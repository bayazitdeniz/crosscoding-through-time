from utils import *

import os
import gc
import glob
import tqdm
from types import SimpleNamespace

from buffer import BufferNNSight
from sae_lens import SAE
from crosscoder import CrossCoder, BatchTopKCrossCoder

torch.set_printoptions(precision=8)

def load_crosscoder(project, checkpoint_number=None, path="./workspace/logs/checkpoints", verbose=True):        
    save_dir = Path(path) / str(project)
    ckpt_version_list = [
        int(currfile.name.split(".")[0])
        for currfile in list(save_dir.iterdir())
        if ".pt" in str(currfile)
    ]
    if checkpoint_number is None:
        if len(ckpt_version_list):
            checkpoint_number = max(ckpt_version_list)
        else:
            raise ValueError("No version found.")
        
    cfg_path = save_dir / f"{str(checkpoint_number)}_cfg.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    
    print(f"Loading checkpoint #{checkpoint_number} from {save_dir}.")
    crosscoder = None
    if "batch_topk_final" in cfg_dict and cfg_dict["batch_topk_final"] is not None:
        print("Loading BatchTopKCrossCoder ...")
        crosscoder = BatchTopKCrossCoder.load(version_dir=project, checkpoint_version=checkpoint_number, path=path, verbose=verbose)
    else:
        print("Loading L1 CrossCoder ...")
        crosscoder = CrossCoder.load(version_dir=project, checkpoint_version=checkpoint_number, path=path, verbose=verbose)
    
    print("Done loading.")
    return crosscoder, cfg_dict, len(ckpt_version_list)

def get_losses(ae, x):
    # x: [batch, n_models, d_model]
    x = x.to(ae.dtype)
    # f: [batch, d_hidden]
    f = ae.encode(x)
    x_hat = ae.decode(f)
    
    e = x_hat.float() - x.float()
    squared_diff = e.pow(2)
    l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
    l2_loss = l2_per_batch.mean()

    total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
    explained_variance = 1 - l2_per_batch / total_variance

    l2_list = []
    explained_variance_list = []
    
    for model_idx in range(ae.cfg["n_models"]):
        per_token_l2_loss_A = (x_hat[:, model_idx, :] - x[:, model_idx, :]).pow(2).sum(dim=-1).squeeze()
        l2_list.append(per_token_l2_loss_A)
        total_variance_A = (x[:, model_idx, :] - x[:, model_idx, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A
        explained_variance_list.append(explained_variance_A)
    
    explained_variance_model_specific = torch.stack(explained_variance_list, dim=0)
    l2_loss_model_specific = torch.stack(l2_list, dim=0)
    
    decoder_norms = ae.W_dec.norm(dim=-1)
    # decoder_norms: [d_hidden, n_models]
    total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
    l1_loss = (f * total_decoder_norm[None, :]).sum(-1).mean(0)

    l0_loss = (f>0).float().sum(-1).mean()

    return SimpleNamespace(**{
        "l2_loss": l2_loss, 
        "l2_loss_model_specific": l2_loss_model_specific, 
        "l1_loss": l1_loss, 
        "l0_loss": l0_loss,
        "explained_variance": explained_variance, 
        "explained_variance_model_specific": explained_variance_model_specific,
    })


def run_sparsity_analysis(cfg, ae_model, model_type, buffer, is_val, num_batches):
    dim = None
    if "dict_size" in cfg.keys():
        dim = cfg["dict_size"]
    elif "d_sae" in cfg.keys():
        dim = cfg["d_sae"]
    
    alive_neuron_count = torch.zeros(dim, dtype=int).to('cuda:0')
    dead_neuron_count = torch.zeros(dim, dtype=int).to('cuda:0')
    
    metric_list = []
    for i in tqdm.trange(num_batches):
        acts = buffer.next()
        
        losses = get_losses(ae_model, acts)
        loss_dict = {
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "explained_variance_mean": losses.explained_variance.mean().item(),
            "explained_variance_std": losses.explained_variance.std().item(),
        }
                
        f = ae_model.encode(acts)
        curr_deads = (f == 0).all(dim=0)
        alive_neuron_count[~curr_deads] += 1
        dead_neuron_count[curr_deads] += 1
        
        for model_idx in range(cfg["n_models"]):
            letter = chr(ord('A') + model_idx)
            loss_dict[f"l2_loss_{letter}"] = losses.l2_loss_model_specific[model_idx].mean().item()
            loss_dict[f"explained_variance_{letter}"] = losses.explained_variance_model_specific[model_idx].mean().item()
        
        metric_list.append(loss_dict)
    
    alive_count = (alive_neuron_count != 0).float().sum().item()
    dead_count = (alive_neuron_count == 0).float().sum().item()
        
    split_name = None
    if is_val:
        split_name = "val"
    else:
        split_name = "train"
        
    stats_dict = avg_std_dictlist(metric_list=metric_list)
    stats_dict["model_type"] = model_type
    stats_dict["ae_dim"] = dim
    stats_dict["data_split"] = split_name
    stats_dict["seed"] = cfg["seed"]
    
    stats_dict["alive_count"] = alive_count
    stats_dict["alive_frac"] = alive_count / cfg["dict_size"]
    stats_dict["dead_count"] = dead_count
    stats_dict["dead_frac"] = dead_count / cfg["dict_size"]
    
    torch.save(dead_neuron_count, os.path.join("workspace/results/sparsity_ce/joint_eval_sparsity", "dead-neuron-count_{}-{}-{}.pt".format(model_type, dim, split_name).replace("/", "_")))
    torch.save(alive_neuron_count, os.path.join("workspace/results/sparsity_ce/joint_eval_sparsity", "alive-neuron-count_{}-{}-{}.pt".format(model_type, dim, split_name).replace("/", "_")))
        
    print("Average & Std Results for {} split and {} batches:".format(split_name, len(metric_list)))
    print(stats_dict)
    
    save_path = os.path.join("workspace/results/sparsity_ce/joint_eval_sparsity", "{}-{}-{}-{}.json".format(model_type, dim, split_name, cfg["seed"]).replace("/", "_"))
    with open(save_path, "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    save_path = os.path.join("workspace/results/sparsity_ce/joint_eval_sparsity", "list_{}-{}-{}-{}.json".format(model_type, dim, split_name, cfg["seed"]).replace("/", "_"))
    with open(save_path, "w") as f:
        json.dump(metric_list, f, indent=4)
        
    return stats_dict
        

@torch.no_grad()
def get_ce_recovered_metrics(tokens, model_list, submod_layer, ae_model, num_batches, batch_size):
    ae_model.eval()
    metric_list = []
    model_name_lower = ae_model.cfg["model_name"].lower()
    
    for i in tqdm.trange(num_batches):
        torch.cuda.empty_cache()
        gc.collect()
        # print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        # print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
        
        batch = tokens[i * batch_size : (i + 1) * batch_size].to("cuda:0")  # Ensure batch is on CUDA
        
        ce_clean_list, ce_zero_abl_list, resid_act_list = [], [], []
        for model in model_list:
            model.eval()
            submodule = None
            if "pythia" in model_name_lower:
                submodule = model.gpt_neox.layers[submod_layer]
            elif "olmo" in model_name_lower:
                submodule = model.model.layers[submod_layer]
            elif "bloom" in model_name_lower:
                submodule = model.transformer.h[submod_layer]
            else:
                raise NotImplementedError("Model name not supported yet.")
            
            # get clean loss & inputs to crosscoder
            with torch.no_grad(), model.trace(batch, **tracer_kwargs): #, invoker_args={}):
                ce_clean = model.output.save()
                resid_act_A = submodule.output.save()
            
            ce_clean_list.append(ce_clean.logits.to("cpu"))
            resid_act_list.append(resid_act_A[0].to("cpu"))
            del ce_clean, resid_act_A

            # get zero abl loss
            with torch.no_grad(), model.trace(batch, **tracer_kwargs): #, invoker_args={}):
                x = submodule.output
                submodule.output[0][:] = torch.zeros_like(x[0])
                logits_zero = model.output.save()
                # restore submodule to original version
                submodule.output = x
            
            ce_zero_abl_list.append(logits_zero.logits.to("cpu"))
            del logits_zero
        
        ae_model_input = torch.stack(resid_act_list, dim=0).to("cuda:0")
        del resid_act_list
        # NOTE: Not skipping BOS tokens
        # ae_model_input = ae_model_input[:, :, 1:, :] # Drop BOS
        ae_model_input = einops.rearrange(
            ae_model_input,
            "n_models batch seq_len d_model -> (batch seq_len) n_models d_model",
        )
        with torch.no_grad():
            ae_model_output = ae_model.decode(ae_model.encode(ae_model_input))
        ae_model_output = einops.rearrange(
            ae_model_output,
            "(batch seq_len) n_models d_model -> n_models batch seq_len d_model", batch = batch.shape[0]
        )

        metrics = {}    
        for model_idx, model in enumerate(model_list):
            model.eval()
            submodule = None
            if "pythia" in model_name_lower:
                submodule = model.gpt_neox.layers[submod_layer]
            elif "olmo" in model_name_lower:
                submodule = model.model.layers[submod_layer]
            elif "bloom" in model_name_lower:
                submodule = model.transformer.h[submod_layer]
            else:
                raise NotImplementedError("Model name not supported yet.")
            letter = chr(ord('A') + model_idx)            
            x_hat = ae_model_output[model_idx]

            # get spliced loss
            with model.trace(batch, **tracer_kwargs):
                submodule.output[0][:] = x_hat
                reconstructed_output = model.output.save()
            
            reconstructed_logits = reconstructed_output.logits
            ce_clean_A_logits = ce_clean_list[model_idx]
            ce_zero_abl_A_logits = ce_zero_abl_list[model_idx]
            
            for name, logits in [
                ("ce_clean_", ce_clean_A_logits), 
                ("ce_loss_spliced_", reconstructed_logits), 
                ("ce_zero_abl_", ce_zero_abl_A_logits), 
            ]:
                logits = logits.to("cuda:0")
                batch = batch.to(logits.device)
                with torch.no_grad():
                    loss = torch.nn.CrossEntropyLoss(**{})(
                        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                        batch[:, 1:].reshape(-1)
                    ).item()
                logits = logits.to("cpu")
                batch = batch.to(logits.device)
                del logits
                metrics[name + letter] = loss
                del loss
            
            metrics["ce_diff_" + letter] = metrics["ce_loss_spliced_" + letter] - metrics["ce_clean_" + letter]
            metrics["ce_recovered_" + letter] = 1 - (metrics["ce_diff_" + letter] / (metrics["ce_zero_abl_" + letter] - metrics["ce_clean_" + letter]))
        
        del batch
        del ce_clean_list
        del ce_zero_abl_list
        metric_list.append(metrics)
    return metric_list


def fold_activation_scaling_factor(ae_model, model_idx, scaling_factor):
    ae_model.W_enc.data[model_idx, :, :] = ae_model.W_enc.data[model_idx, :, :] * scaling_factor
    ae_model.W_dec.data[:, model_idx, :] = ae_model.W_dec.data[:, model_idx, :] / scaling_factor
    ae_model.b_dec.data[model_idx, :] = ae_model.b_dec.data[model_idx, :] / scaling_factor
    return ae_model

def splice_act_hook(act, hook, spliced_act):
    act[:, 1:, :] = spliced_act # Drop BOS
    return act

def zero_ablation_hook(act, hook):
    act[:] = 0
    return act

def run_ce_analysis(cfg, model_type, tokens, ae_model, model_list, is_val, num_batches, batch_size):
    ae_model = ae_model #.to(torch.bfloat16)
    submod_layer = int(cfg["hook_point"].split(".")[1]) - 1
    metric_list = get_ce_recovered_metrics(tokens, model_list, submod_layer, ae_model, num_batches=num_batches, batch_size=batch_size)
    
    dim = None
    if "dict_size" in cfg.keys():
        dim = cfg["dict_size"]
    elif "d_sae" in cfg.keys():
        dim = cfg["d_sae"]
        
    split_name = None
    if is_val:
        split_name = "val"
    else:
        split_name = "train"
        
    stats_dict = avg_std_dictlist(metric_list)
    stats_dict["model_type"] = model_type
    stats_dict["ae_dim"] = dim
    stats_dict["data_split"] = split_name
    stats_dict["seed"] = cfg["seed"]
    
    print("Average & Std Results for {} split and {} batches of size {}:".format(split_name, num_batches, batch_size))
    print(stats_dict)
    
    
    save_path = os.path.join("joint_eval_ce", "{}-{}-{}-{}.json".format(model_type, dim, split_name, cfg["seed"]).replace("/", "_"))
    with open(save_path, "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    save_path = os.path.join("joint_eval_ce", "list_{}-{}-{}-{}.json".format(model_type, dim, split_name, cfg["seed"]).replace("/", "_"))
    with open(save_path, "w") as f:
        json.dump(metric_list, f, indent=4)
        
    return stats_dict


def get_args(description='arguments for test_any.py main func', jupyter=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--project_path",
        type=str,
        default="./workspace/logs/checkpoints",
        help="...")
    parser.add_argument(
        "--project_name",
        type=str,
        help="version_0")
    parser.add_argument(
        "--do_ce_eval",
        action='store_true', 
        default=True,
        help="...")
    parser.add_argument(
        "--do_sparsity_eval",
        action='store_true', 
        default=False,
        help="...")
    parser.add_argument(
        "--last_ckpt_only",
        action='store_true', 
        default=False,
        help="...")
    args = None

    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()

    return args


def main():
    ############################################################################
    # 1) Load AE according to argparse
    ############################################################################
    model_list = None
    estimated_act_norm_list = None
    cfg = None
    
    # get autoencoder path
    args = get_args()
    device = 'cuda:0'
        
    # load autoencoder
    ae_model, cfg, ckpnt_list_len = load_crosscoder(project=args.project_name, path=args.project_path)
        
    print("-" * 60)
    print("Run info:")
    print("MODEL TYPE: ", cfg["model_type"])
    print("MODEL NAME: ", cfg["model_name"])
    print("-" * 60)
    
    ############################################################################
    # 2) Load model according to AE config
    ############################################################################
    seed = 4352
    set_seed(seed)
    device = 'cuda:0'
    
    model_list = load_revision_nnsight(
        model_name=cfg["model_name"],
        revision_list=cfg["revision_list"].split("_") if "olmo" in cfg["model_name"].lower() else cfg["revision_list"].split("-"),
        device=device,
        seed=cfg["seed"]
    )
    
    step_model_names = []
    train_tokens = None
    val_tokens = None
    if "olmo" in cfg["model_name"].lower():
        step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("_")]
        train_tokens = load_olmo_dolma_filtered(is_val=False)
        val_tokens = load_olmo_dolma_filtered(is_val=True)
    elif "pythia" in cfg["model_name"].lower():
        step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("-")]
        train_tokens = load_pile_pythia_filtered(is_val=False)
        val_tokens = load_pile_pythia_filtered(is_val=True)
    elif "bloom" in cfg["model_name"].lower():
        step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("-")]
        train_tokens = load_bloom_c4(is_val=False)
        val_tokens = load_bloom_c4(is_val=True)
    else:
        raise NotImplementedError("Model name not supported yet.")
    
    wandb.init(project="crosscoder-eval", entity=cfg["wandb_entity"], name=cfg["wandb_name"].replace("train-", ""), config=cfg)
    
    loop_ckpt_list = list(range(ckpnt_list_len))
    if args.last_ckpt_only:
        loop_ckpt_list = loop_ckpt_list[-1:]
        
    for i in loop_ckpt_list:
        ae_model, cfg, _ = load_crosscoder(project=args.project_name, checkpoint_number=i, path=args.project_path)
        
        if "estimated_act_norm_list" in cfg:
            estimated_act_norm_list = cfg["estimated_act_norm_list"]
        else:
            estimated_act_norm_list = [scaling_factor_dict[cfg["hook_point"]][step][0] for step in step_model_names]
        print("Estimated norm factor list: ", estimated_act_norm_list)
        
        step = (i * cfg["save_every"]) + 1
        print(f"Evaluating on checkpoint for step: {step}")
        
        batch_size = 8
        num_batches = 30
        
        if args.do_sparsity_eval:
            ############################################################################
            # 3) Load the train data and run analysis
            ############################################################################
            buffer = BufferNNSight(cfg, model_list, train_tokens, estimated_act_norm_list, do_shuffle=False)
            stats_dict = run_sparsity_analysis(
                cfg=cfg,
                ae_model=ae_model,
                model_type=cfg["model_type"],
                buffer=buffer,
                is_val=False,
                num_batches=num_batches
            )
            wandb.log({f"train/{key}": value for key, value in stats_dict.items()}, step=step)

            
            ############################################################################
            # 4) Load the val data and run analysis
            ############################################################################
            buffer = BufferNNSight(cfg, model_list, val_tokens, estimated_act_norm_list, do_shuffle=False)
            stats_dict = run_sparsity_analysis(
                cfg=cfg,
                ae_model=ae_model,
                model_type=cfg["model_type"],
                buffer=buffer,
                is_val=True,
                num_batches=num_batches
            )
            wandb.log({f"val/{key}": value for key, value in stats_dict.items()}, step=step)
        
        if args.do_ce_eval:
            ############################################################################
            # 3) Load the train data and run analysis
            ############################################################################
            for model_idx, scaling_factor in enumerate(estimated_act_norm_list):
                ae_model = fold_activation_scaling_factor(ae_model, model_idx, scaling_factor)
            
            stats_dict = run_ce_analysis(
                cfg=cfg,
                model_type=cfg["model_type"],
                tokens=train_tokens,
                ae_model=ae_model,
                model_list=model_list,
                is_val=False,
                batch_size=batch_size,
                num_batches=num_batches
            )
            wandb.log({f"train/{key}": value for key, value in stats_dict.items()}, step=step)
            
            ############################################################################
            # 4) Load the val data and run analysis
            ############################################################################
            stats_dict = run_ce_analysis(
                cfg=cfg,
                model_type=cfg["model_type"],
                tokens=val_tokens,
                ae_model=ae_model,
                model_list=model_list,
                is_val=True,
                batch_size=batch_size,
                num_batches=num_batches
            )
            wandb.log({f"val/{key}": value for key, value in stats_dict.items()}, step=step)


if __name__ == '__main__':
    main()
