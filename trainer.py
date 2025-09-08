from utils import *
from crosscoder import CrossCoder, BatchTopKCrossCoder
from buffer import BufferNNSight
import tqdm
import time 
import datetime

from torch.nn.utils import clip_grad_norm_
from types import SimpleNamespace

def format_time(seconds):
    """Convert seconds to a human readable string in the format HH:MM:SS"""
    return str(datetime.timedelta(seconds=int(round(seconds))))

def fold_activation_scaling_factor(ae_model, model_idx, scaling_factor):
    ae_model.W_enc.data[model_idx, :, :] = ae_model.W_enc.data[model_idx, :, :] * scaling_factor
    ae_model.W_dec.data[:, model_idx, :] = ae_model.W_dec.data[:, model_idx, :] / scaling_factor
    ae_model.b_dec.data[model_idx, :] = ae_model.b_dec.data[model_idx, :] / scaling_factor
    return ae_model

def unfold_activation_scaling_factor(ae_model, model_idx, scaling_factor):
    ae_model.W_enc.data[model_idx, :, :] = ae_model.W_enc.data[model_idx, :, :] / scaling_factor
    ae_model.W_dec.data[:, model_idx, :] = ae_model.W_dec.data[:, model_idx, :] * scaling_factor
    ae_model.b_dec.data[model_idx, :] = ae_model.b_dec.data[model_idx, :] * scaling_factor
    return ae_model

def splice_act_hook(act, hook, spliced_act):
    act[:, 1:, :] = spliced_act # Drop BOS
    return act

def zero_ablation_hook(act, hook):
    act[:] = 0
    return act


class TrainerNNSight:
    def __init__(self, cfg, model_list, train_tokens, val_tokens):
        self.cfg = cfg
        self.model_list = model_list
        assert len(model_list) == self.cfg["n_models"]
        for model in model_list:
            model.eval()
        self.crosscoder = CrossCoder(cfg)
        
        self.estimated_act_norm_list = None
        try:
            if cfg["model_type"] in scaling_factor_dict:
                self.estimated_act_norm_list = scaling_factor_dict[cfg["model_type"]]
            else:
                if "pythia" in cfg["model_name"].lower() or "bloom" in cfg["model_name"].lower():
                    step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("-")]
                    self.estimated_act_norm_list = [scaling_factor_dict[step][0] for step in step_model_names]
                elif "olmo" in cfg["model_name"].lower():
                    step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("_")]
                    self.estimated_act_norm_list = [scaling_factor_dict[step][0] for step in step_model_names]
                else:
                    raise ValueError("Norm scaling not in dict for this type of model yet.")
        except:
            print("Norm scaling factor is not cached for this crosscoder type. We let the buffer calculate from the given train dataset...")
            self.estimated_act_norm_list = None
        
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        
        self.train_buffer = BufferNNSight(cfg, model_list, train_tokens, self.estimated_act_norm_list, do_shuffle=True)
        self.val_buffer = BufferNNSight(cfg, model_list, val_tokens, self.estimated_act_norm_list, do_shuffle=False)
        
        self.estimated_act_norm_list = self.train_buffer.normalisation_factor.tolist()
        self.cfg["estimated_act_norm_list"] = self.train_buffer.normalisation_factor.tolist()
        self.crosscoder.cfg["estimated_act_norm_list"] = self.train_buffer.normalisation_factor.tolist()
        
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]
        
        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.step_counter = 0

        wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"], name=cfg["wandb_name"], config=cfg)

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < self.cfg["l1_warmup_pct"] * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (self.cfg["l1_warmup_pct"] * self.total_steps)
        else:
            return self.cfg["l1_coeff"]
        
    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        x = x.to(self.crosscoder.dtype)
        f = self.crosscoder.encode(x)
        # f: [batch, d_hidden]
        x_hat = self.crosscoder.decode(f)
        e = x_hat.float() - x.float()
        squared_diff = e.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        explained_variance = 1 - l2_per_batch / total_variance

        l2_list = []
        explained_variance_list = []
        
        for model_idx in range(self.crosscoder.cfg["n_models"]):
            per_token_l2_loss_A = (x_hat[:, model_idx, :] - x[:, model_idx, :]).pow(2).sum(dim=-1).squeeze()
            l2_list.append(per_token_l2_loss_A)
            total_variance_A = (x[:, model_idx, :] - x[:, model_idx, :].mean(0)).pow(2).sum(-1).squeeze()
            explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A
            explained_variance_list.append(explained_variance_A)
        
        explained_variance_model_specific = torch.stack(explained_variance_list, dim=0)
        l2_loss_model_specific = torch.stack(l2_list, dim=0)

        decoder_norms = self.crosscoder.W_dec.norm(dim=-1)
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
            "explained_variance_model_specific": explained_variance_model_specific
        })

    def train_step(self):
        self.crosscoder.train()
        acts = self.train_buffer.next()
        losses = self.get_losses(acts)
        # print(losses.l2_loss)
        # print(losses.l1_loss)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        # print(loss)
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
        }
        
        for model_idx in range(self.cfg["n_models"]):
            letter = chr(ord('A') + model_idx)
            loss_dict[f"l2_loss_{letter}"] = losses.l2_loss_model_specific[model_idx].mean().item()
            loss_dict[f"explained_variance_{letter}"] = losses.explained_variance_model_specific[model_idx].mean().item()

        self.step_counter += 1
        train_loss_dict = {f"train/{key}": value for key, value in loss_dict.items()}
        return train_loss_dict
    
    @torch.no_grad()
    def val_step(self):
        self.crosscoder.eval()
        self.val_buffer.restart()
        val_losses = []
        num_val_batches = self.cfg["num_val_batches"]
        
        for _ in range(num_val_batches):
            acts = self.val_buffer.next()
            losses = self.get_losses(acts)
            loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
            val_loss_dict = {
                "loss": loss.item(),
                "l2_loss": losses.l2_loss.item(),
                "l1_loss": losses.l1_loss.item(),
                "l0_loss": losses.l0_loss.item(),
                "explained_variance": losses.explained_variance.mean().item(),
            }
            for model_idx in range(self.cfg["n_models"]):
                letter = chr(ord('A') + model_idx)
                val_loss_dict[f"l2_loss_{letter}"] = losses.l2_loss_model_specific[model_idx].mean().item()
                val_loss_dict[f"explained_variance_{letter}"] = losses.explained_variance_model_specific[model_idx].mean().item()
            val_losses.append(val_loss_dict)

        # Average validation metrics across the batches.
        aggregated_loss = {}
        for key in val_losses[0]:
            aggregated_loss[f"val/{key}"] = sum(d[key] for d in val_losses) / len(val_losses)
        
        self.crosscoder.train()
        return aggregated_loss

    def log(self, loss_dict, verbose=False):
        wandb.log(loss_dict, step=self.step_counter)
        if verbose:
            print(loss_dict)

    def save(self):
        self.crosscoder.save()

    def train(self, do_ce_eval):
        self.step_counter = 0
        # cntr = 0.0
        try:
            for i in tqdm.trange(self.total_steps):
                self.crosscoder.train()
                # start_train = time.time()
                loss_dict = self.train_step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                # train_elapsed = time.time() - start_train
                # cntr += train_elapsed
                # print("train_elapsed: ", train_elapsed)
                if i % self.cfg["save_every"] == 0:
                    self.save()
                # sparsity metrics
                if i % self.cfg["val_log_every"] == 0:
                    # start_val = time.time()
                    val_loss_dict = self.val_step()
                    self.log(val_loss_dict)
                # ce metrics
                if do_ce_eval and (i % (self.cfg["val_log_every"] * 4) == 0):
                    for split_name, split_tokens in [("train", self.train_tokens), ("val", self.val_tokens)]:
                        ce_metrics = self.get_ce_recovered_metrics(
                            split_tokens, 
                            self.model_list, 
                            self.crosscoder
                        )
                        stats_dict = avg_std_dictlist(ce_metrics)
                        stats_dict = {f"{split_name}/{key}": value for key, value in stats_dict.items()}
                        self.log(stats_dict)
                    # val_elapsed = time.time() - start_val
                    # cntr += val_elapsed
                    # print("val_elapsed: ", val_elapsed)
                    # if i > 0:
                    #     avg_per_loop = cntr / i
                    #     estimated_total = avg_per_loop * self.total_steps
                    #     print("#" * 80)
                    #     print("#" * 80)
                    #     print("time so far: ", format_time(cntr))
                    #     print("each loop avg:", format_time(avg_per_loop))
                    #     print("curr estimated time:", format_time(estimated_total))
                    #     print("#" * 80)
                    #     print("#" * 80)
        finally:
            self.save()


class BatchTopKTrainerNNSight(TrainerNNSight):
    """
    Code mostly from: https://github.com/science-of-finetuning/dictionary_learning/blob/main/dictionary_learning/trainers/crosscoder.py
    """
    def __init__(self, cfg, model_list, train_tokens, val_tokens):
        self.cfg = cfg
        self.model_list = model_list
        assert len(model_list) == self.cfg["n_models"]
        for model in model_list:
            model.eval()
                
        self.crosscoder = BatchTopKCrossCoder(cfg)
        
        self.top_k_aux = cfg["d_in"] // 2  # Heuristic from B.1 of the BatchTopKCrossCoder paper
        
        self.num_tokens_since_fired_train = torch.zeros(cfg["dict_size"], dtype=torch.long, device=self.cfg["device"])
        self.num_tokens_since_fired_val = torch.zeros(cfg["dict_size"], dtype=torch.long, device=self.cfg["device"])
        self.running_deads_train =  -1
        self.running_deads_val =  -1
        
        self.estimated_act_norm_list = None
        try:
            if cfg["model_type"] in scaling_factor_dict:
                self.estimated_act_norm_list = scaling_factor_dict[cfg["model_type"]]
            else:
                if "pythia" in cfg["model_name"].lower() or "bloom" in cfg["model_name"].lower():
                    step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("-")]
                    self.estimated_act_norm_list = [scaling_factor_dict[step][0] for step in step_model_names]
                elif "olmo" in cfg["model_name"].lower():
                    step_model_names = [cfg["model_name"] + "/" + step for step in cfg["revision_list"].split("_")]
                    self.estimated_act_norm_list = [scaling_factor_dict[step][0] for step in step_model_names]
                else:
                    raise ValueError("Norm scaling not in dict for this type of model yet.")
        except:
            print("Norm scaling factor is not cached for this crosscoder type. We let the buffer calculate from the given train dataset...")
            self.estimated_act_norm_list = None
        
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        
        self.train_buffer = BufferNNSight(cfg, model_list, train_tokens, self.estimated_act_norm_list, do_shuffle=True)
        self.val_buffer = BufferNNSight(cfg, model_list, val_tokens, self.estimated_act_norm_list, do_shuffle=False)
        
        self.estimated_act_norm_list = self.train_buffer.normalisation_factor.tolist()
        self.cfg["estimated_act_norm_list"] = self.train_buffer.normalisation_factor.tolist()
        self.crosscoder.cfg["estimated_act_norm_list"] = self.train_buffer.normalisation_factor.tolist()
        
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]
        
        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.step_counter = 0

        wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"], name=cfg["wandb_name"], config=cfg)

    def update_k(self):
        """Annealing function for k."""
        if self.cfg["topk_annealing_steps"] > 0 and self.cfg["batch_topk_init"] != self.cfg["batch_topk_final"]:
            if self.step_counter < self.cfg["topk_annealing_steps"]:
                progress = float(self.step_counter) / self.cfg["topk_annealing_steps"]
                # Linear interpolation from k_initial down to k_target
                current_k_float = (
                    self.cfg["batch_topk_init"] - (self.cfg["batch_topk_init"] - self.cfg["batch_topk_final"]) * progress
                )
                new_k_val = max(1, int(round(current_k_float)))
                if self.crosscoder.k.item() != new_k_val:
                    self.crosscoder.k.fill_(new_k_val)
            else:  # Annealing finished, ensure k is set to k_target
                if self.crosscoder.k.item() != self.cfg["batch_topk_final"]:
                    self.crosscoder.k.fill_(self.cfg["batch_topk_final"])
        elif (
            self.cfg["topk_annealing_steps"] == 0 and self.crosscoder.k.item() != self.cfg["batch_topk_init"]
        ):
            # If no annealing steps, k should be fixed at k_initial
            self.crosscoder.k.fill_(self.cfg["batch_topk_init"])

    def update_threshold(self, f_scaled):
        """Update the activation threshold using exponential moving average."""
        # f/f_scaled: [batch, d_hidden]
        active = f_scaled[f_scaled > 0]
        if active.size(0) == 0:
            min_activation = 0.0
        else:
            min_activation = active.min().detach().to(dtype=torch.float32)

        if self.crosscoder.threshold < 0:
            self.crosscoder.threshold = min_activation
        else:
            self.crosscoder.threshold = (self.cfg["threshold_beta"] * self.crosscoder.threshold) + (
                (1 - self.cfg["threshold_beta"]) * min_activation
            )
            
    def get_auxiliary_loss(
        self,
        e: torch.Tensor,
        post_relu_f: torch.Tensor,
        post_relu_f_scaled: torch.Tensor,
        is_val=False
    ):
        """
        Compute auxiliary loss to resurrect dead features.

        This implements an auxk loss similar to TopK and BatchTopKSAE that attempts
        to make dead latents active again by computing reconstruction loss on the
        top-k dead features.

        Args:
            e: Residual tensor (batch_size, num_layers, model_dim)
            post_relu_f: Post-ReLU feature activations
            post_relu_f_scaled: Scaled post-ReLU feature activations

        Returns:
            Normalized auxiliary loss tensor
        """
        # x: [batch, n_models, d_model]
        batch_size, num_layers, model_dim = e.size()
        # Reshape to (batch_size, n_models*d_model)
        e = e.reshape(batch_size, -1)
        dead_features = None
        if is_val:
            dead_features = self.num_tokens_since_fired_val >= self.cfg["dead_feature_threshold"]
            self.running_deads_val = int(dead_features.sum())
        else:
            dead_features = self.num_tokens_since_fired_train >= self.cfg["dead_feature_threshold"]
            self.running_deads_train = int(dead_features.sum())

        if dead_features.sum() > 0:
            # If there are dead features the loss is not 0.0
            
            # Choose how many to resurrect
            k_aux = min(self.top_k_aux, dead_features.sum())

            # Mask and detach scaled activations:
            # This sets all non-dead entries to -inf, 
            # so the top-k will only come from the dead ones.
            # Indexing with None broadcasts the mask to (1, num_features)
            # so you can broadcast to (batch_size, num_features)
            auxk_latents_scaled = torch.where(
                dead_features[None], post_relu_f_scaled, -torch.inf
            ).detach()

            # Find the top-k dead feature ids
            _, auxk_indices = auxk_latents_scaled.topk(k_aux, sorted=False)
            
            # Build a sparse activation buffer
            auxk_buffer_BF = torch.zeros_like(post_relu_f)
            row_indices = (
                torch.arange(post_relu_f.size(0), device=post_relu_f.device)
                .view(-1, 1)
                .expand(-1, auxk_indices.size(1))
            )
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=post_relu_f[row_indices, auxk_indices]
            )

            # Reconstruct via the decoder
            #   NOTE: we don't want to apply the bias
            x_hat_aux = self.crosscoder.decode(auxk_acts_BF, apply_bias=False)
            x_hat_aux = x_hat_aux.reshape(batch_size, -1)
            
            # Compute L2 reconstruction loss
            l2_loss_aux = (
                (e.float() - x_hat_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )
            # Stored for logging
            pre_norm_auxk_loss = l2_loss_aux.item()

            # Normalize the loss
            #   NOTE: normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = e.mean(dim=0)[None, :].broadcast_to(
                e.shape
            )
            loss_denom = (
                (e.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            # pre_norm_auxk_loss returned for logging
            return normalized_auxk_loss.nan_to_num(0.0), pre_norm_auxk_loss
        else:
            # If no dead features the loss is 0.0
            return torch.tensor(0, dtype=e.dtype, device=e.device), -1

    
    def get_losses(self, x, is_val):
        # x: [batch, n_models, d_model]
        x = x.to(self.crosscoder.dtype)
        self.update_k()
        
        f, f_scaled, active_indices_F, post_relu_f, post_relu_f_scaled = self.crosscoder.encode(x, return_all=True)
        
        if self.step_counter > self.cfg["threshold_start_step"]:
            self.update_threshold(f_scaled)
        
        # f: [batch, d_hidden]
        x_hat = self.crosscoder.decode(f)
                    
        fired_counter = (
            self.num_tokens_since_fired_val
            if is_val
            else self.num_tokens_since_fired_train
        )
        did_fire = torch.zeros_like(fired_counter, dtype=torch.bool)
        did_fire[active_indices_F] = True
        num_tokens_in_step = x.size(0)
        fired_counter += num_tokens_in_step
        fired_counter[did_fire] = 0
        
        e = x_hat.float() - x.float()
        squared_diff = e.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        explained_variance = 1 - l2_per_batch / total_variance

        l2_list = []
        explained_variance_list = []
        
        for model_idx in range(self.crosscoder.cfg["n_models"]):
            per_token_l2_loss_A = (x_hat[:, model_idx, :] - x[:, model_idx, :]).pow(2).sum(dim=-1).squeeze()
            l2_list.append(per_token_l2_loss_A)
            total_variance_A = (x[:, model_idx, :] - x[:, model_idx, :].mean(0)).pow(2).sum(-1).squeeze()
            explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A
            explained_variance_list.append(explained_variance_A)
        
        explained_variance_model_specific = torch.stack(explained_variance_list, dim=0)
        l2_loss_model_specific = torch.stack(l2_list, dim=0)

        decoder_norms = self.crosscoder.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
        l1_loss = (f * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_counts = (f>0).float().sum(dim=-1)
        l0_loss = l0_counts.mean()
        l0_loss_std = l0_counts.std()
        
        # NOTE: doing e.detach()
        auxk_loss, pre_norm_auxk_loss = self.get_auxiliary_loss(
            e=e.detach(),
            post_relu_f=post_relu_f,
            post_relu_f_scaled=post_relu_f_scaled,
            is_val=is_val
        )
        
        return SimpleNamespace(**{
            "l2_loss": l2_loss, 
            "l2_loss_model_specific": l2_loss_model_specific, 
            "l1_loss": l1_loss, 
            "l0_loss": l0_loss,
            "l0_loss_std": l0_loss_std,
            "explained_variance": explained_variance, 
            "explained_variance_model_specific": explained_variance_model_specific,
            "auxk_loss": auxk_loss,
            "pre_norm_auxk_loss": pre_norm_auxk_loss
        })

    def train_step(self):
        self.crosscoder.train()
        acts = self.train_buffer.next()
        losses = self.get_losses(acts, is_val=False)
        # print(losses.l2_loss)
        # print(losses.auxk_loss)
        loss = losses.l2_loss + self.cfg["auxk_alpha"] * losses.auxk_loss
        # print(loss)
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l0_loss_std": losses.l0_loss_std.item(),
            "auxk_loss": losses.auxk_loss.item(),
            "pre_norm_auxk_loss": losses.pre_norm_auxk_loss,
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "k": self.crosscoder.k.item(),
            "threshold": self.crosscoder.threshold.item(),
            "running_deads": self.running_deads_train
        } 
        
        for model_idx in range(self.cfg["n_models"]):
            letter = chr(ord('A') + model_idx)
            loss_dict[f"l2_loss_{letter}"] = losses.l2_loss_model_specific[model_idx].mean().item()
            loss_dict[f"explained_variance_{letter}"] = losses.explained_variance_model_specific[model_idx].mean().item()

        self.step_counter += 1
        train_loss_dict = {f"train/{key}": value for key, value in loss_dict.items()}
        return train_loss_dict
    
    @torch.no_grad()
    def val_step(self):
        self.crosscoder.eval()
        self.val_buffer.restart()
        val_losses = []
        num_val_batches = self.cfg["num_val_batches"]
        
        for _ in range(num_val_batches):
            acts = self.val_buffer.next()
            losses = self.get_losses(acts, is_val=True)
            loss = losses.l2_loss + self.cfg["auxk_alpha"] * losses.auxk_loss
            val_loss_dict = {
                "loss": loss.item(),
                "l2_loss": losses.l2_loss.item(),
                "l1_loss": losses.l1_loss.item(),
                "l0_loss": losses.l0_loss.item(),
                "l0_loss_std": losses.l0_loss_std.item(),
                "auxk_loss": losses.auxk_loss.item(),
                "pre_norm_auxk_loss": losses.pre_norm_auxk_loss,
                "explained_variance": losses.explained_variance.mean().item(),
                "running_deads": self.running_deads_val
            }
            for model_idx in range(self.cfg["n_models"]):
                letter = chr(ord('A') + model_idx)
                val_loss_dict[f"l2_loss_{letter}"] = losses.l2_loss_model_specific[model_idx].mean().item()
                val_loss_dict[f"explained_variance_{letter}"] = losses.explained_variance_model_specific[model_idx].mean().item()
            val_losses.append(val_loss_dict)

        # Average validation metrics across the batches.
        aggregated_loss = {}
        for key in val_losses[0]:
            aggregated_loss[f"val/{key}"] = sum(d[key] for d in val_losses) / len(val_losses)
        
        self.crosscoder.train()
        return aggregated_loss
