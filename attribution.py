from dataclasses import dataclass
from nnsight.intervention import Envoy
from collections import namedtuple
import torch
import einops
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Any

import gc
import os
import argparse
import csv
from vis import (
    calc_rel_dec_norm,
    calc_rel_ie,
    save_feature_vis,
    revision2tokens_pythia,
    revision2tokens_bloom,
    revision2tokens_olmo,
)

import copy
from utils import set_seed, load_revision_nnsight, scaling_factor_dict
from utils import (
    load_pile_pythia_filtered,
    load_olmo_dolma_filtered,
    load_bloom_c4_uniform_multiblimp,
    load_task_dataset
)
from test_crosscoder_revision import fold_activation_scaling_factor, load_crosscoder

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])
torch.autograd.set_detect_anomaly(True)


class SparseAct:
    """
    A SparseAct is a helper class which represents a vector in the sparse feature basis provided by an SAE, jointly with the SAE error term.
    A SparseAct may have three fields:
    act : the feature activations in the sparse basis
    res : the SAE error term
    resc : a contracted SAE error term, useful for when we want one number per feature and error (instead of having d_model numbers per error)
    """

    def __init__(
            self, 
            act: torch.Tensor,
            res: torch.Tensor | None = None,
            resc: torch.Tensor | None = None, # contracted residual
        ) -> None:

            self.act = act
            self.res = res
            self.resc = resc

    def _map(self, f, aux=None) -> 'SparseAct':
        kwargs = {}
        if isinstance(aux, SparseAct):
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return SparseAct(**kwargs)

    def __mul__(self, other) -> 'SparseAct':
        return self._map(lambda x, y: x * y, other)

    def __rmul__(self, other) -> 'SparseAct':
        # This will handle float/int * SparseAct by reusing the __mul__ logic
        return self.__mul__(other)

    def __matmul__(self, other: 'SparseAct') -> 'SparseAct':
        assert self.res is not None and other.res is not None
        # dot product between two SparseActs, except only the residual is contracted
        return SparseAct(act = self.act * other.act, resc=(self.res * other.res).sum(dim=-1, keepdim=True))
    
    def __add__(self, other) -> 'SparseAct':
        return self._map(lambda x, y: x + y, other)

    def __radd__(self, other: 'SparseAct') -> 'SparseAct':
        return self.__add__(other)
    
    def __sub__(self, other: 'SparseAct') -> 'SparseAct':
        return self._map(lambda x, y: x - y, other)
    
    def __truediv__(self, other) -> 'SparseAct':
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return SparseAct(**kwargs)

    def __rtruediv__(self, other) -> 'SparseAct':
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return SparseAct(**kwargs)

    def __neg__(self) -> 'SparseAct':
        return self._map(lambda x, _: -x)
    
    def __invert__(self) -> 'SparseAct':
            return self._map(lambda x, _: ~x)
    
    def __getitem__(self, index: int):
        return self.act[index]
    
    def __repr__(self):
        if self.res is None:
            return f"SparseAct(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"SparseAct(act={self.act}, res={self.res})"
        else:
            raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")
    
    def sum(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return SparseAct(**kwargs)
    
    def mean(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return SparseAct(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return SparseAct(**kwargs)
    
    def clone(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return SparseAct(**kwargs)
    
    @property
    def value(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            v = getattr(self, attribute)
            if v is not None:
                # if it has a .value, unwrap it; otherwise leave it as-is
                kwargs[attribute] = v.value if hasattr(v, 'value') else v
        return SparseAct(**kwargs)

    def save(self):
        return self._map(lambda x, _: x.save())
    
    def detach(self):
        return self._map(lambda x, _: x.detach())
    
    def to_tensor(self):
        if self.resc is None:
            assert self.res is not None
            return torch.cat([self.act, self.res], dim=-1)
        if self.res is None:
            assert self.resc is not None
            return torch.cat([self.act, self.resc], dim=-1)
        raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")

    def to(self, device):
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    def __eq__(self, other): # type: ignore
        return self._map(lambda x, y: x == y, other)
    
    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)
    
    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)
    
    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())
    
    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))
    
    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)
    
    def zeros_like(self):
        return self._map(lambda x, _: torch.zeros_like(x))
    
    def ones_like(self):
        return self._map(lambda x, _: torch.ones_like(x))
    
    def abs(self):
        return self._map(lambda x, _: x.abs())

@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Envoy
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self):
        return hash(self.name)

    def get_activation(self):
        out = self.submodule.input if self.use_input else self.submodule.output
        return out[0] if self.is_tuple else out

    def set_activation(self, x):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0][:] = x
            else:
                self.submodule.input[:] = x
        else:
            if self.is_tuple:
                self.submodule.output[0][:] = x
            else:
                self.submodule.output[:] = x

    def stop_grad(self):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0].grad = torch.zeros_like(self.submodule.input[0])
            else:
                self.submodule.input.grad = torch.zeros_like(self.submodule.input)
        else:
            if self.is_tuple:
                self.submodule.output[0].grad = torch.zeros_like(self.submodule.output[0])
            else:
                self.submodule.output.grad = torch.zeros_like(self.submodule.output)


def get_crosscoder_activation(
    model_input,
    model_list,
    submodule_list,
    dictionary,
    metric_fn,
    metric_kwargs
):
    """
    Assumes one submodule per model.
    Returns:
        hidden_states_list: List of dicts mapping each submodule to its SparseAct
        metrics_list: List of metric snapshots for each model
    """
    # 1. Collect activations from all models
    activations = []
    for model, submodule in zip(model_list, submodule_list):
        with torch.no_grad(), model.trace(model_input):
            activations.append(submodule.get_activation().save())

    # 2. Stack activations: (n_models, batch, seq_len, d_model)
    x = torch.stack(activations, dim=0).to("cuda:0")
    n_models, batch, seq_len, d_model = x.shape

    # 3. Format for cross-coding: (batch*seq_len, n_models, d_model)
    x_format = einops.rearrange(
        x, "n_models batch seq_len d_model -> (batch seq_len) n_models d_model"
    )

    # 4. Encode and decode through dictionary
    with torch.no_grad():
        f = dictionary.encode(x_format)
        x_hat = dictionary.decode(f)

    # 5. Reshape back to (n_models, batch, seq_len, d_model)
    x_hat = einops.rearrange(
        x_hat,
        "(batch seq_len) n_models d_model -> n_models batch seq_len d_model",
        batch=batch,
    )
    
    # 6. Validate reconstruction shapes
    for orig, recon in zip(activations, x_hat):
        assert orig.shape == recon.shape, \
            f"Shape mismatch: {orig.shape} vs {recon.shape}"

    # 7. Compute hidden states and metrics per model
    hidden_states_list = []
    metrics_list = []
    for model, submodule, orig, recon in zip(
        model_list, submodule_list, activations, x_hat
    ):
        hidden_states = {}
        with torch.no_grad(), model.trace(model_input):
            residual = orig - recon
            hidden_states = {
                submodule: SparseAct(act=f, res=residual.save())
            }
            metrics = metric_fn(model, **metric_kwargs).save()

        hidden_states = {k: v.value for k, v in hidden_states.items()}
        hidden_states_list.append(hidden_states)
        metrics_list.append(metrics)
    
    return hidden_states_list, metrics_list


def get_accumulated_gradients_for_steps(
    model,
    submodule,
    dictionary,
    hidden_states_clean,
    hidden_states_patch,
    clean,
    steps,
    model_idx,
    metric_fn,
    metric_kwargs
):
    """
    Compute path-integrated gradients between clean and patch SparseActs at a submodule.
    Assumes one submodule per model.
    """
    effects = {}
    deltas = {}
    grads = {}
    
    clean_state = hidden_states_clean[submodule]
    patch_state = hidden_states_patch[submodule]
    with model.trace() as tracer:
        metrics = []
        fs = []
        for step in range(steps):
            alpha = step / steps
            f = (1 - alpha) * clean_state + alpha * patch_state
            f.act.requires_grad_().retain_grad()
            f.res.requires_grad_().retain_grad()
            fs.append(f)
            with tracer.invoke(clean):
                x_hat = dictionary.decode(f.act)
                x_hat = einops.rearrange(
                    x_hat,
                    "(batch seq_len) n_models d_model -> n_models batch seq_len d_model",
                    batch=len(clean),
                )
                submodule.set_activation(x_hat[model_idx] + f.res)
                metrics.append(metric_fn(model, **metric_kwargs))
        metric = sum([m for m in metrics])
        metric.sum().backward()
    
    mean_grad = sum([f.act.grad for f in fs]) / steps
    mean_residual_grad = sum([f.res.grad for f in fs]) / steps
    grad = SparseAct(act=mean_grad, res=mean_residual_grad)  # type: ignore
    delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
    effect = grad @ delta

    effects[submodule] = effect
    deltas[submodule] = delta
    grads[submodule] = grad
    
    return effects, deltas, grads


def _pe_ig(
    clean,
    patch,
    model_list,
    submodule_list, # list[Submodule],
    dictionary, #: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
):
    """
    Generalized path integrated gradients over multiple models.

    Returns:
        List of EffectOut for each model.
    """
    # Clean activations and metrics
    clean_states, clean_metrics = get_crosscoder_activation(
        model_input=clean,
        model_list=model_list,
        submodule_list=submodule_list,
        dictionary=dictionary,
        metric_fn=metric_fn,
        metric_kwargs=metric_kwargs
    )

    # Patch activations and total effects
    patch_states = []
    total_effects = []
    if patch is None:
        for state in clean_states:
            patch_states.append({
                k: SparseAct(act=torch.zeros_like(v.act), res=torch.zeros_like(v.res))
                for k, v in state.items()
            })
            total_effects.append(None)
    else:
        patch_states, patch_metrics = get_crosscoder_activation(
            model_input=patch,
            model_list=model_list,
            submodule_list=submodule_list,
            dictionary=dictionary,
            metric_fn=metric_fn,
            metric_kwargs=metric_kwargs
        )
        total_effects = [
            (p - c).detach()
            for p, c in zip(patch_metrics, clean_metrics)
        ]

    # Accumulate gradients & effects for each model
    results = []
    for idx, (model, submodule, clean_state, patch_state, total_eff) in enumerate(
        zip(model_list, submodule_list, clean_states, patch_states, total_effects)
    ):
        effects, deltas, grads = get_accumulated_gradients_for_steps(
            model=model,
            submodule=submodule,
            dictionary=dictionary,
            hidden_states_clean=clean_state,
            hidden_states_patch=patch_state,
            clean=clean,
            steps=steps,
            model_idx=idx,
            metric_fn=metric_fn,
            metric_kwargs=metric_kwargs
        )
        results.append(EffectOut(effects, deltas, grads, total_eff))

    return results


def get_nodes(
    dataset,
    dataset_name,
    model_list,
    submodule_list,
    dictionary,
    node_threshold=0.1,
    do_threshold: bool = False,
    aggregation=None,
    max_examples=1000,
    batch_size=4,
    base_dir="./workspace/logs/ie_dicts_zeroshot",
    version_num=0,
    ckpt_num=0,
    device="cuda:0"
):
    """
    Compute per-feature nodes/effects over a dataset and save to disk.
    """
    num_examples = min(len(dataset), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [dataset[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    if num_examples < max_examples:
        print(f"Total examples < {max_examples}, using {num_examples}.")

    running_nodes = None

    print(batches[0])
    print(batches[1])
    
    for batch in tqdm(batches, desc="Batches"):
        clean_inputs = batch["clean_prefix"]
        clean_answer_idxs = torch.tensor(
            model_list[0].tokenizer(batch["clean_answer"]).input_ids,
            dtype=torch.long,
            device=device
        ).squeeze(-1)

        if "patch_prefix" not in batch and "patch_answer" not in batch:
            patch_inputs = None
            patch_answer_idxs = None
            def metric_fn(model):
                return -1 * torch.gather(
                    model.output.logits[:, -1, :],
                    dim=-1,
                    index=clean_answer_idxs.view(-1, 1),
                ).squeeze(-1)
        elif "patch_prefix" not in batch and "patch_answer" in batch:
            patch_inputs = None
            patch_answer_idxs = torch.tensor(
                model_list[0].tokenizer(batch["patch_answer"]).input_ids,
                dtype=torch.long,
                device=device,
            )
            def metric_fn(model):
                logits = model.output.logits[:, -1, :]
                return torch.gather(
                    logits, dim=-1, index=patch_answer_idxs.view(-1, 1)
                ).squeeze(-1) - torch.gather(
                    logits, dim=-1, index=clean_answer_idxs.view(-1, 1)
                ).squeeze(-1)
        else:
            patch_inputs = batch["patch_prefix"]
            patch_answer_idxs = torch.tensor(
                model_list[0].tokenizer(batch["patch_answer"]).input_ids,
                dtype=torch.long,
                device=device,
            )
            def metric_fn(model):
                logits = model.output.logits[:, -1, :]
                return torch.gather(
                    logits, dim=-1, index=patch_answer_idxs.view(-1, 1)
                ).squeeze(-1) - torch.gather(
                    logits, dim=-1, index=clean_answer_idxs.view(-1, 1)
                ).squeeze(-1)
        
        # Compute integrated gradients for all models
        effect_outs = _pe_ig(
            clean_inputs,
            patch_inputs,
            model_list,
            submodule_list,
            dictionary,
            metric_fn,
            steps=10,
            metric_kwargs={}
        )

        # Collect node activities
        nodes = {}
        for idx, eff in enumerate(effect_outs):
            if eff.total_effect is not None:
                nodes[f"y_m{idx}"] = list(eff.total_effect.values())[0]
            for sub, act_res in eff.effects.items():
                key = sub.name
                assert key.startswith("m") and key[1].isdigit(), \
                    f"sub.name should start with 'm' followed by a digit, got '{sub.name}'"
                nodes[key] = act_res.act
                if do_threshold:
                    mask = nodes[key].abs() > node_threshold
                    nodes[key] = nodes[key] * mask

        # Aggregation & mean over batch
        if aggregation == "sum":
            for k in list(nodes.keys()):
                if not k.startswith("y_m"):
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}

        # Accumulate
        if running_nodes is None:
            running_nodes = {k: v.to("cpu") * len(batch) for k, v in nodes.items() if not k.startswith("y_m")}
        else:
            for k, v in nodes.items():
                if not k.startswith("y_m"):
                    running_nodes[k] += v.to("cpu") * len(batch)

        # Memory cleanup
        del nodes
        gc.collect()

    save_dict = {k: (v.to(device) / num_examples) for k, v in running_nodes.items()}
    save_dir = os.path.join(base_dir, str(version_num))
    os.makedirs(save_dir, exist_ok=True)
    thresh_str = None if not do_threshold else str(node_threshold)
    final_path = os.path.join(save_dir, f"{dataset_name}_ckpt{ckpt_num}_thresh{thresh_str}_n{num_examples}.pt")

    torch.save(save_dict, final_path)
    return running_nodes, save_dict


def get_split_output(
    model,
    submodule,
    dictionary,
    hidden_states_clean,
    hidden_states_patch,
    clean,
    model_idx,
    metric_fn,
    metric_kwargs
):
    """
    Forward once with patched SparseAct at a single submodule and return metric.
    Assumes one submodule per model.
    """
    clean_state = hidden_states_clean[submodule]
    patch_state = hidden_states_patch[submodule]
    with model.trace() as tracer:
        f = patch_state
        f.act.requires_grad_().retain_grad()
        f.res.requires_grad_().retain_grad()
        with tracer.invoke(clean):
            x_hat = dictionary.decode(f.act)
            x_hat = einops.rearrange(
                x_hat,
                "(batch seq_len) n_models d_model -> n_models batch seq_len d_model", batch=len(clean)
            )
            submodule.set_activation(x_hat[model_idx] + f.res)
            metric = metric_fn(model, **metric_kwargs).save()
    
    return metric


def compute_token_probs_with_feature_ablation(
    clean_prefix,
    model_list,
    submodule_list, # list[Submodule],
    dictionary, #: dict[Submodule, Dictionary],
    metric_fn,
    ablate_feat_id: int,
    metric_kwargs=dict()
):
    """
    Compute metric values before/after ablating a single feature id for each model.
    """
    if ablate_feat_id is None:
        raise ValueError(f"Ablated feature ID needs to be an int and cannot be {ablate_feat_id}")
    
    # 1) Get clean activations
    m1_hidden_clean, m2_hidden_clean, m1_metrics_clean, m2_metrics_clean = get_crosscoder_activation(
        model_input=clean_prefix,
        model_list=model_list,
        submodule_list=submodule_list,
        dictionary=dictionary,
        metric_fn=metric_fn,
        metric_kwargs=metric_kwargs
    )

    # 2) Get patch activations per model
    hidden_clean_list = [m1_hidden_clean, m2_hidden_clean]
    hidden_patch_list = []
    for hidden_clean in hidden_clean_list:
        hidden_patch = {}
        for k, v in hidden_clean.items():
            new_act = v.act.clone()
            # new_act shape: [batch_size, dict_size]
            new_act[:, ablate_feat_id] = 0.0
            hidden_patch[k] = SparseAct(
                act=new_act,
                res=v.res.clone()
            )
        hidden_patch_list.append(hidden_patch)
    
    # 3) Get probabilities of splitted forwards per model
    metrics_patch = []
    for idx in range(len(model_list)):
        metrics_patch.append(
            get_split_output(
                model=model_list[idx],
                submodule=submodule_list[idx],
                dictionary=dictionary,
                hidden_states_clean=hidden_clean_list[idx],
                hidden_states_patch=hidden_patch_list[idx],
                clean=clean_prefix,
                model_idx=idx,
                metric_fn=metric_fn,
                metric_kwargs=metric_kwargs
            )
        )
    
    return (m1_metrics_clean.tolist(), m2_metrics_clean.tolist(), metrics_patch[0].tolist(), metrics_patch[1].tolist())


def get_prob_diffs(
    dataset,
    dataset_name,
    model_list,
    submodule_list,
    dictionary,
    ablate_feat_dict_list,
    max_examples=1000,
    batch_size=4,
    base_dir="./workspace/logs/ie_dicts_zeroshot",
    version_num=None,
    top_k=None,
    device="cuda:0"
):
    """
    Summarize precomputed ablation results from CSVs and write summary + correlations.
    """
    # 1) Prepare batches
    # num_examples = min(len(dataset), max_examples)
    # n_batches = math.ceil(num_examples / batch_size)
    # batches = [
    #     dataset[batch * batch_size : (batch + 1) * batch_size]
    #     for batch in range(n_batches)
    # ]
    # if num_examples < max_examples:  # warn the user
    #     print(
    #         f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead."
    #     )
    # print("Example batch #0: ", batches[0])
    # print("Example batch #1: ", batches[1])
    
    # # 2) Collect probs for all feat_ids for correct and incorrect answer
    # #    when activations are clean (original) and patched (zeroed out at the id)
    # def compute_metrics_for_batch(batch, model_list, submodule_list, dictionary, feat_id, device, answer_type):
    #     """
    #     Helper func
    #     """
    #     inputs = batch["clean_prefix"]
    #     answer_tokens = torch.tensor(
    #         model_list[0].tokenizer(batch[f"{answer_type}_answer"]).input_ids,
    #         dtype=torch.long,
    #         device=device
    #     ).squeeze(-1)

    #     def metric_fn(model):
    #         last_logits = model.output.logits[:, -1, :]
    #         last_logprobs = F.log_softmax(last_logits, dim=-1)
    #         return last_logprobs.gather(dim=-1, index=answer_tokens.view(-1, 1)).squeeze(-1)

    #     return compute_token_probs_with_feature_ablation(
    #         inputs,
    #         model_list,
    #         submodule_list,
    #         dictionary,
    #         metric_fn,
    #         feat_id
    #     )
    
    # indiv_rows = []
    
    # for feat_dict in ablate_feat_dict_list:
    #     for batch in tqdm(batches, desc="Batches"):
    #         # clean=correct, patch=wrong
    #         correct_m1_clean, correct_m2_clean, correct_m1_patch, correct_m2_patch = compute_metrics_for_batch(
    #             batch, model_list, submodule_list, dictionary, feat_dict["feat_id"], device, "clean"
    #         )
    #         wrong_m1_clean, wrong_m2_clean, wrong_m1_patch, wrong_m2_patch = compute_metrics_for_batch(
    #             batch, model_list, submodule_list, dictionary, feat_dict["feat_id"], device, "patch"
    #         )
    #         for cm1c, cm2c, cm1p, cm2p, wm1c, wm2c, wm1p, wm2p in zip(
    #             correct_m1_clean, correct_m2_clean, correct_m1_patch, correct_m2_patch,
    #             wrong_m1_clean, wrong_m2_clean, wrong_m1_patch, wrong_m2_patch
    #         ):
    #             indiv_rows.append({
    #                 "version": feat_dict["version"],
    #                 "comparison": feat_dict["comparison"],
    #                 "feat_id": feat_dict["feat_id"],
    #                 "task": feat_dict["task"],
    #                 "RelDec": feat_dict["rel_dec_norm_value"],
    #                 "RelIE": feat_dict["rel_ie_value"],
    #                 "corr_m1_clean":  cm1c,
    #                 "corr_m2_clean":  cm2c,
    #                 "corr_m1_patch":  cm1p,
    #                 "corr_m2_patch":  cm2p,
    #                 "wrong_m1_clean": wm1c,
    #                 "wrong_m2_clean": wm2c,
    #                 "wrong_m1_patch": wm1p,
    #                 "wrong_m2_patch": wm2p,
    #             })
            
    #         gc.collect()
            
    # # 3) Save DataFrame for individual results
    # indiv_df = pd.DataFrame(indiv_rows)
    # save_dir = f"{base_dir}/{version_num}"
    # # if not os.path.exists(save_dir):
    # #     os.makedirs(save_dir)
    # if len(ablate_feat_dict_list) == 1:
    #     indiv_path = f"{save_dir}/ablation-task_{dataset_name}-featid_{feat_id}-probdiffs.csv"
    #     indiv_df.to_csv(indiv_path, index=False)
    # else:
    #     indiv_path = f"{save_dir}/ablation-task_{dataset_name}-topk{top_k}-probdiffs.csv"
    #     indiv_df.to_csv(indiv_path, index=False)
    
    save_dir = f"{base_dir}/{version_num}"
    indiv_path = f"{save_dir}/ablation-task_{dataset_name}-topk100-probdiffs.csv"
    indiv_df = pd.read_csv(indiv_path)
    
    # select only the rows whose feat_id is in the ablate_feat_dict_list i.e. top10 if that's what's passed
    topkfeats = set([feat_dict["feat_id"] for feat_dict in ablate_feat_dict_list])
    print("# of rows before: ", len(indiv_df))
    indiv_df = indiv_df[indiv_df["feat_id"].isin(topkfeats)]
    print("# of rows after: ", len(indiv_df))

        
    # 4) Save DataFrame for summary of results
    summary_rows = []
    group_cols = ["version", "comparison", "feat_id", "task", "RelDec", "RelIE"]
    
    fid2relie2 = {feat_dict["feat_id"]: feat_dict["rel_ie2_value"] for feat_dict in ablate_feat_dict_list}
    
    for (ver, comparison, fid, task, rel_dec, rel_ie), group in indiv_df.groupby(group_cols):
        # raw means
        m1_clean_acc = (group["corr_m1_clean"] > group["wrong_m1_clean"]).mean()
        m1_patch_acc = (group["corr_m1_patch"] > group["wrong_m1_patch"]).mean()
        m2_clean_acc = (group["corr_m2_clean"] > group["wrong_m2_clean"]).mean()
        m2_patch_acc = (group["corr_m2_patch"] > group["wrong_m2_patch"]).mean()
        
        m1_clean_logprobdiff = (group["corr_m1_clean"] - group["wrong_m1_clean"]).mean()
        m1_patch_logprobdiff = (group["corr_m1_patch"] - group["wrong_m1_patch"]).mean()
        m2_clean_logprobdiff = (group["corr_m2_clean"] - group["wrong_m2_clean"]).mean()
        m2_patch_logprobdiff = (group["corr_m2_patch"] - group["wrong_m2_patch"]).mean()

        # percentages & deltas
        m1_clean_acc_pct  = m1_clean_acc * 100
        m2_clean_acc_pct  = m2_clean_acc * 100
        m1_patch_acc_delta = (m1_clean_acc - m1_patch_acc) * 100
        m2_patch_acc_delta = (m2_clean_acc - m2_patch_acc) * 100
        
        m1_patch_logprobdiff_delta = m1_clean_logprobdiff - m1_patch_logprobdiff
        m2_patch_logprobdiff_delta = m2_clean_logprobdiff - m2_patch_logprobdiff

        record = {
            "version":              ver,
            "comparison":           comparison,
            "feat_id":              fid,
            "task":                 task,
            "RelDec":               rel_dec,
            "RelIE":                rel_ie,
            "RelIE2":               fid2relie2[fid],
            #
            "M1 Clean Acc (%)":     m1_clean_acc_pct,
            "M1 Patch Δ (pp)":      m1_patch_acc_delta,
            "M2 Clean Acc (%)":     m2_clean_acc_pct,
            "M2 Patch Δ (pp)":      m2_patch_acc_delta,
            #
            "M1 Clean LogProbDiff":     m1_clean_logprobdiff,
            "M1 Patch LogProbDiff Δ":   m1_patch_logprobdiff_delta,
            "M2 Clean LogProbDiff":     m2_clean_logprobdiff,
            "M2 Patch LogProbDiff Δ":   m2_patch_logprobdiff_delta,
            #
            "Acc Δ Ratio (M1/M2)":  (m1_patch_acc_delta + 1e-16) / (m2_patch_acc_delta + 1e-16),
            #
            "LogProbDiff Δ Ratio (M1/M2)":  (m1_patch_logprobdiff_delta + 1e-16) / (m2_patch_logprobdiff_delta + 1e-16),
            "LogProbDiff Δ Ratio (M2/M1)":  (m2_patch_logprobdiff_delta + 1e-16) / (m1_patch_logprobdiff_delta + 1e-16),
            #
            "LogProbDiff Δ Ratio Abs(M1/M2)":  abs(m1_patch_logprobdiff_delta + 1e-16) / abs(m2_patch_logprobdiff_delta + 1e-16),
            "LogProbDiff Δ Ratio Abs(M2/M1)":  abs(m2_patch_logprobdiff_delta + 1e-16) / abs(m1_patch_logprobdiff_delta + 1e-16),
            "Normalized LogProbDiff Δ Ratio (M2 - M1 / M1)": (abs(m2_patch_logprobdiff_delta + 1e-16) - abs(m1_patch_logprobdiff_delta + 1e-16)) / abs(m1_patch_logprobdiff_delta + 1e-16),
        }
        summary_rows.append(record)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = None
    
    if len(ablate_feat_dict_list) == 1:
        summary_path = f"{save_dir}/ablation-task_{dataset_name}-featid_{feat_id}-deltasummary.csv"
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_path = f"{save_dir}/ablation-task_{dataset_name}-topk{top_k}-deltasummary.csv"
        summary_df.to_csv(summary_path, index=False)
    
    print(f"Saved summary results to {summary_path}")
    
    # 5) Get correlations across all features
    if len(ablate_feat_dict_list) != 1:
        first_dict = ablate_feat_dict_list[0]
        corr_dict = {
            "comparison": first_dict["comparison"],
            "task": first_dict["task"],
            "rho(Δ Ratio (M1/M2) - RelDec)": summary_df["LogProbDiff Δ Ratio (M1/M2)"].corr(summary_df["RelDec"], method="spearman"),
            "rho(Δ Ratio (M1/M2) - RelIE)": summary_df["LogProbDiff Δ Ratio (M1/M2)"].corr(summary_df["RelIE"], method="spearman"),
            #
            "rho(Δ Ratio Abs(M1/M2) - RelDec)": summary_df["LogProbDiff Δ Ratio Abs(M1/M2)"].corr(summary_df["RelDec"], method="spearman"),
            "rho(Δ Ratio Abs(M1/M2) - RelIE)": summary_df["LogProbDiff Δ Ratio Abs(M1/M2)"].corr(summary_df["RelIE"], method="spearman"),
            #
            "rho(Δ Ratio (M2/M1) - RelDec)": summary_df["LogProbDiff Δ Ratio (M2/M1)"].corr(summary_df["RelDec"], method="spearman"),
            "rho(Δ Ratio (M2/M1) - RelIE)": summary_df["LogProbDiff Δ Ratio (M2/M1)"].corr(summary_df["RelIE"], method="spearman"),
            #
            "rho(Δ Ratio Abs(M2/M1) - RelDec)": summary_df["LogProbDiff Δ Ratio Abs(M2/M1)"].corr(summary_df["RelDec"], method="spearman"),
            "rho(Δ Ratio Abs(M2/M1) - RelIE)": summary_df["LogProbDiff Δ Ratio Abs(M2/M1)"].corr(summary_df["RelIE"], method="spearman"),
        }    
        print(corr_dict)
        corr_df = pd.DataFrame([corr_dict])
        corr_save_path = f"{save_dir}/ablation-task_{dataset_name}-topk{top_k}-corr.csv"
        corr_df.to_csv(corr_save_path, index=False)
        print(f"Saved correlation results to {corr_save_path}")


def build_submodule_list(
    model_list: List[Any],
    model_name_lowered: str,
    layer_idx: int
) -> List[Any]:
    """
    Constructs a list of Submodule wrappers for each model's specified layer output.

    Args:
        model_list: models to hook into (e.g., different variants).
        hook_point: string of form "module.<layer_index>.out" or similar to extract layer index.
        name_prefix: prefix for naming each submodule (default 'm', becomes 'm0_', 'm1_', etc.).

    Returns:
        submodule_list: List of Submodule(name, submodule, use_input=False, is_tuple=True).
    """
    # Extract layer index from hook_point, e.g., 'h.3.out' -> 3
    submodule_list = []
    for idx, model in enumerate(model_list):
        sub = None
        if "pythia" in model_name_lowered:
            sub = model.gpt_neox.layers[layer_idx]
        elif "olmo" in model_name_lowered:
            sub = model.model.layers[layer_idx]
        elif "bloom" in model_name_lowered:
            sub = model.transformer.h[layer_idx]
        else:
            raise NotImplementedError(f"Model '{model_name_lowered}' not supported yet.")

        # name each submodule with its model index and layer
        submodule_name = f"m{idx}_layer{layer_idx}_out"
        print(submodule_name)
        submodule_list.append(
            Submodule(
                name=submodule_name,
                submodule=sub,
                use_input=False,
                is_tuple=True
            )
        )
    return submodule_list


def get_args(description='arguments for attribution.py main func', jupyter=False):
    from utils import str2bool
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--project_path",
        type=str,
        default="./workspace/logs/checkpoints",
        help="...")
    parser.add_argument(
        "--version_num",
        type=str,
        help="e.g. version_0")
    parser.add_argument(
        "--ckpt_num",
        type=str,
        default="20")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="e.g. hellaswag or anaphorgender")
    parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.1)
    parser.add_argument(
        "--do_threshold",
        type=str2bool)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=1000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4)
    parser.add_argument(
        "--just_ablation",
        type=str2bool,
        default=False)
    parser.add_argument(
        "--indiv_feat_ablation",
        type=str2bool)
    parser.add_argument(
        "--ablate_feat_id",
        type=int,
        default=None)
    parser.add_argument(
        "--ie_precomputed",
        type=str2bool)
    parser.add_argument(
        "--vis_with_lmodeling",
        type=str2bool)
    parser.add_argument(
        "--html_saved",
        type=str2bool)
    parser.add_argument(
        "--serve_html",
        type=str2bool,
        default=False)
    parser.add_argument(
        "--top_k",
        type=int,
        default=10)
    parser.add_argument(
        "--bottom_and_top",
        type=str2bool,
        default=False)
    parser.add_argument(
        "--port",
        type=int,
        default=8084)

    if jupyter:
        return parser.parse_args('')
    return parser.parse_args()


def main():
    """
    Entry point: load models + crosscoder, compute IE, save csv/viz, optionally ablate.
    """
    set_seed(42)
    args = get_args()
    
    ############################################################################
    # (1) Load all relevant, model, crosscoder, data
    ############################################################################
    
    crosscoder, _, _ = load_crosscoder(
        project=args.version_num,
        checkpoint_number=args.ckpt_num,
        path=args.project_path,
        verbose=False
    )

    model_name_lowered = crosscoder.cfg["model_name"].lower()
    rev_str = crosscoder.cfg["revision_list"]
    rev_list = rev_str.split("_") if "olmo" in model_name_lowered else rev_str.split("-")
    model_list = load_revision_nnsight(
        model_name=crosscoder.cfg["model_name"],
        revision_list=rev_list,
        device="cuda:0",
        seed=crosscoder.cfg["seed"]
    )
    step_model_names = [crosscoder.cfg["model_name"] + "/" + step for step in rev_list]

    folded_crosscoder = copy.deepcopy(crosscoder)

    if "estimated_act_norm_list" in folded_crosscoder.cfg:
        estimated_act_norm_list = folded_crosscoder.cfg["estimated_act_norm_list"]
    else:
        estimated_act_norm_list = [scaling_factor_dict[folded_crosscoder.cfg["hook_point"]][step][0] for step in step_model_names]

    for model_idx, scaling_factor in enumerate(estimated_act_norm_list):
        folded_crosscoder = fold_activation_scaling_factor(folded_crosscoder, model_idx, scaling_factor)

    submod_layer = int(folded_crosscoder.cfg["hook_point"].split(".")[1]) - 1
    submodule_list = build_submodule_list(
        model_list=model_list,
        model_name_lowered=model_name_lowered,
        layer_idx=submod_layer
    )
    
    dataset = load_task_dataset(args.dataset_name, tokenizer=model_list[0].tokenizer)
    max_example_task_list = ["anaphor", "subjectverb", "clams", "multiblimp"]
    if any([args.dataset_name.startswith(task) for task in max_example_task_list]):
        args.max_examples = len(dataset["train"])
    
    ############################################################################
    # (2) Compute IE
    ############################################################################
    
    if not args.ie_precomputed:
        _, effects = get_nodes(
            dataset=dataset["train"],
            dataset_name=args.dataset_name,
            model_list=model_list,
            submodule_list=submodule_list,
            dictionary=folded_crosscoder,
            node_threshold=args.node_threshold,
            do_threshold=args.do_threshold,
            aggregation=None,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
            base_dir="./workspace/logs/ie_dicts_zeroshot",
            version_num=args.version_num,
            ckpt_num=args.ckpt_num,
            device="cuda:0"
        )
    else:
        base_dir = "./workspace/logs/ie_dicts_zeroshot"
        save_dir = f"{base_dir}/{args.version_num}"
        final_path = f"{save_dir}/{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}.pt"
        effects = torch.load(final_path)
    
    
    ############################################################################
    # (3) Create CSV so you can annotate the features
    #     feat_id
    #     task
    #     comparison (can get from crosscoder cfg)
    #     seed (can get from crosscoder cfg)
    #     task_category (if hellaswag, winogrande, arc challenge than "functional", otherwise "formal")
    #     selection_source (default to "ie" for now)
    #     activation_dataset (default to "task")
    #     rel_dec_norm_value ()
    #     ie_value [A B]
    #     rel_dec_norm_class
    #     ie_class
    ############################################################################

    abs_ies = []
    ie_keys = list(effects.keys())
    ie_keys.sort()
    print("Sorted IE keys: ", ie_keys)
    for k in ie_keys:
        abs_ies.append(effects[k].cpu().abs())
    abs_ie_array = np.array(abs_ies)
    assert len(abs_ie_array) == len(model_list), "Number of models does not match the number of IE arrays"

    latent_list = []
    for ie in abs_ie_array:
        non_zero_count = (ie != 0.0).sum()
        temp_top_k = args.top_k if args.top_k <= non_zero_count else non_zero_count
        if not args.bottom_and_top:
            model_latent_ies = set(torch.topk(torch.tensor(ie), k=temp_top_k).indices.tolist())
        else:
            # get bottom & top, not just top
            model_latent_ies = set(
                torch.topk(torch.tensor(ie), k=temp_top_k).indices.tolist() +
                torch.topk(torch.tensor(ie), k=temp_top_k, largest=False).indices.tolist()
            )
        latent_list.append(model_latent_ies)
    
    # find intersection of all elements in latent_list for any size
    shared_latent_ids = latent_list[0]
    latents_to_study = latent_list[0]
    for latent_set in latent_list[1:]:
        shared_latent_ids = shared_latent_ids.intersection(latent_set)
        latents_to_study = latents_to_study.union(latent_set)

    print("Feats to study (union): ", len(latents_to_study))
    print("Shared feats (intersection): ", len(shared_latent_ids))
    if len(model_list) == 3:
        ab_overlap = latent_list[0].intersection(latent_list[1])
        bc_overlap = latent_list[1].intersection(latent_list[2])
        ac_overlap = latent_list[0].intersection(latent_list[2])
        print("Shared feats across A-B: ", len(ab_overlap))
        print(ab_overlap)
        print("Shared feats across B-C: ", len(bc_overlap))
        print(bc_overlap)
        print("Shared feats across A-C: ", len(ac_overlap))
        print(ac_overlap)

    for i in range(len(latent_list)):
        model_latent_ids = latent_list[i]
        print(f"# of model {i} specific feats: ", len(model_latent_ids - shared_latent_ids))

    rel_norms = calc_rel_dec_norm(crosscoder)
    rel_ie, all_zero_count = calc_rel_ie(abs_ie_array, remove_nonzero=False)

    if len(model_list) == 3:
        rel_norms_pair = calc_rel_dec_norm(crosscoder, one_vs_all=False)
        rel_ie_pair, _ = calc_rel_ie(abs_ie_array, remove_nonzero=False, one_vs_all=False)

    def classify_cont_value(value):
        if value > 0.7:
            return 1.0
        if value < 0.3:
            return 0.0
        else:
            return 0.5

    task_category = "functional" if args.dataset_name in ["hellaswag", "winogrande", "arcchallenge"] else "formal"
    
    token_stage = None
    if "pythia" in model_name_lowered:
        token_stage = [revision2tokens_pythia(step) for step in crosscoder.cfg["revision_list"].split("-")]
    elif "bloom" in model_name_lowered:
        token_stage = [revision2tokens_bloom(step) for step in crosscoder.cfg["revision_list"].split("-")]
    elif "olmo" in model_name_lowered:
        token_stage = [revision2tokens_olmo(step) for step in crosscoder.cfg["revision_list"].split("_")]
    else:
        raise NotImplementedError("Model name not supported yet.")
    token_stage_str = " vs. ".join(token_stage)
    
    feat_dict_list = []
    for feat_id in latents_to_study:
        row = {
            "feat_id": feat_id,
            "task": args.dataset_name,
            "layer": submod_layer,
            "version": args.version_num,
            "comparison": token_stage_str,
            "seed": crosscoder.cfg["seed"],
            "task_category": task_category,
            "selection_source": "ie",
            "activation_dataset": "task"
        }
        
        if len(model_list) == 2:
            rel_dec_norm_value = rel_norms[feat_id]
            rel_ie_value = rel_ie[feat_id]
            row.update({
                "ie_value_A": effects[ie_keys[0]][feat_id].item(),
                "ie_value_B": effects[ie_keys[1]][feat_id].item(),
                "rel_dec_norm_value": rel_dec_norm_value,
                "rel_dec_norm_class": classify_cont_value(rel_dec_norm_value),
                "rel_ie_value": rel_ie_value,
                "rel_ie_class": classify_cont_value(rel_ie_value),
            })

        elif len(model_list) == 3:
            rel_dec_norm_value = rel_norms[:, feat_id]
            rel_ie_value = rel_ie[:, feat_id]
            rel_dec_norm_value_pair = rel_norms_pair[:, feat_id]
            rel_ie_value_pair = rel_ie_pair[:, feat_id]

            row.update({
                "ie_value_A": effects[ie_keys[0]][feat_id].item(),
                "ie_value_B": effects[ie_keys[1]][feat_id].item(),
                "ie_value_C": effects[ie_keys[2]][feat_id].item(),
                "rel_dec_norm_value": rel_dec_norm_value,
                "rel_dec_norm_class": [classify_cont_value(norm) for norm in rel_dec_norm_value],
                "rel_dec_norm_value_pair": rel_dec_norm_value_pair,
                "rel_dec_norm_class_pair": [classify_cont_value(norm) for norm in rel_dec_norm_value_pair],
                "rel_ie_value": rel_ie_value,
                "rel_ie_class": [classify_cont_value(norm) for norm in rel_ie_value],
                "rel_ie_value_pair": rel_ie_value_pair,
                "rel_ie_class_pair": [classify_cont_value(norm) for norm in rel_ie_value_pair],
            })

        feat_dict_list.append(row)
        
    # Save feat_dict_list to CSV
    base_dir = "./workspace/logs/ie_dicts_zeroshot"
    save_dir = f"{base_dir}/{args.version_num}"
    bottom_and_top_str = f"_bottomNtop{args.bottom_and_top}" if args.bottom_and_top else ""
    output_path = f"{save_dir}/latents_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}_topk{args.top_k}{bottom_and_top_str}.csv"

    with open(output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=feat_dict_list[0].keys())
        writer.writeheader()
        writer.writerows(feat_dict_list)

    print(f"Saved latent analysis results to {output_path}")
    
    ############################################################################
    # (4) Ablation as an individual feat or group of topk features
    ############################################################################
    if args.just_ablation and args.indiv_feat_ablation:
        get_prob_diffs(
            dataset=dataset["train"],
            dataset_name=args.dataset_name,
            model_list=model_list,
            submodule_list=submodule_list,
            dictionary=folded_crosscoder,
            ablate_feat_dict_list=[args.ablate_feat_id],
            max_examples=args.max_examples,
            batch_size=args.batch_size,
            base_dir="./workspace/logs/ie_dicts_zeroshot",
            version_num=args.version_num,
            top_k=None,
            device="cuda:0"
        )
        exit()

    if args.just_ablation and not args.indiv_feat_ablation:
        get_prob_diffs(
            dataset=dataset["train"],
            dataset_name=args.dataset_name,
            model_list=model_list,
            submodule_list=submodule_list,
            dictionary=folded_crosscoder,
            ablate_feat_dict_list=feat_dict_list,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
            base_dir="./workspace/logs/ie_dicts_zeroshot",
            version_num=args.version_num,
            top_k=args.top_k,
            device="cuda:0"
        )
        exit()
    
    ############################################################################
    # (5) Save the HTML file of the feature visualizations at the same directory 
    #     so you can easily serve them later
    ############################################################################
    # 1) Tokenize the dataset / process it to get inputs and attention
    if "SV-#" in args.dataset_name:
        args.dataset_name = args.dataset_name.replace("SV-#", "SV-N")
    filename = f"{save_dir}/viz_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}_topk{args.top_k}{bottom_and_top_str}.html"
    if not args.html_saved:
        if "test" in dataset:
            del dataset["test"]
        tokenizer = model_list[0].tokenizer

        if not args.vis_with_lmodeling:
            def preprocess_function(examples, max_length):
                # Combine input and target as a completion task
                texts = [inp + tgt for inp, tgt in zip(examples["clean_prefix"], examples["clean_answer"])]
                inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding="max_length", padding_side="right", max_length=max_length)
                return {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            
            # Tokenize each one without padding and get max length
            print(dataset.column_names)
            texts = [inp + tgt for inp, tgt in zip(dataset["train"]["clean_prefix"], dataset["train"]["clean_answer"])]
            tokenized = [tokenizer(t, truncation=False, add_special_tokens=True)["input_ids"] for t in texts]
            max_len = max(len(t) for t in tokenized)
            if args.dataset_name == "anaphor":
                max_len = 16
            print(f"Maximum found length for dataset {args.dataset_name} is {max_len}.")

            tokenized_dataset = dataset.map(lambda x: preprocess_function(x, max_length=max_len), batched=True, remove_columns=dataset["train"].column_names)
            data_split = tokenized_dataset["train"]
            data_split.set_format(type="torch", columns=["input_ids", "attention_mask"])
            all_tokens = data_split["input_ids"].to("cuda:0")
            all_attn = data_split["attention_mask"].to("cuda:0")
        else:
            set_seed(42)
            all_tokens = None
            if "pythia" in model_name_lowered:
                all_tokens = load_pile_pythia_filtered(is_val=False)
            elif "olmo" in model_name_lowered:
                all_tokens = load_olmo_dolma_filtered(is_val=False)
            elif "bloom" in model_name_lowered:
                # all_tokens = load_bloom_c4(is_val=False)
                all_tokens = load_bloom_c4_uniform_multiblimp()
            else:
                raise NotImplementedError("Model name not supported yet.")
            shuffled_indices = torch.randperm(all_tokens.size(0))
            all_tokens = all_tokens[shuffled_indices].to("cuda:0")
            all_attn = torch.ones_like(all_tokens).to("cuda:0")
                    
        # 2) Create & save the viz
        # folded_crosscoder = folded_crosscoder.to("cpu")
        save_feature_vis(
            model_list=model_list,
            folded_crosscoder=folded_crosscoder,
            latents_to_study=latents_to_study,
            tokens=all_tokens,
            attn=all_attn,
            num_examples=500,
            device="cuda:0",
            filename=filename
        )
    
    if args.serve_html:
        from vis import serve_html_file
        serve_html_file(filename, port=args.port)
        
        
    ############################################################################
    # (6) Plot IE vs. Rel Dec figures
    ############################################################################
    if len(model_list) == 2:
        from vis import ie_rel_dec_confusion_matrix, rel_ie_histogram, plot_sorted_distributions, interp_histogram, relative_corr, interp_histogram_by_ie_class
        plot_sorted_distributions(
            dist1=effects[ie_keys[0]].cpu(),
            dist2=effects[ie_keys[1]].cpu(),
            save_path=f"{save_dir}/sorteddist_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}.png"
        )
        
        rel_ie_histogram(
            m1_effects=effects[ie_keys[0]].cpu(), 
            m2_effects=effects[ie_keys[1]].cpu(), 
            label1="M1", 
            label2="M2", 
            title=f"{args.dataset_name} - Relative IE Histogram", 
            save_path=f"{save_dir}/reliehist_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}.png"
        )
        
        df = pd.read_csv(output_path)
        assert len(df["topk_ie_class"]) == len(df["rel_ie_class"])
        ie_rel_dec_confusion_matrix(
            df=df,
            ie_column="topk_ie_class",
            top_k=args.top_k,
            save_path=f"{save_dir}/confmatrixTopKIE_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}_topk{args.top_k}{bottom_and_top_str}.png"
        )
        ie_rel_dec_confusion_matrix(
            df=df,
            ie_column="rel_ie_class",
            top_k=args.top_k,
            save_path=f"{save_dir}/confmatrixRelIE_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}_topk{args.top_k}{bottom_and_top_str}.png"
        )
        
        relative_corr(
            df=df,
            save_path=f"{save_dir}/corrRelIERelDec_{args.dataset_name}_ckpt{args.ckpt_num}_thresh{args.node_threshold}_n{args.max_examples}_topk{args.top_k}{bottom_and_top_str}.png"
        )
        

if __name__ == '__main__':
    main()