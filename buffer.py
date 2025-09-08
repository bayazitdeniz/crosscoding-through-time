from utils import *
from transformer_lens import ActivationCache
import tqdm


class BufferNNSight:
    # Code initially from: https://github.com/ckkissane/crosscoder-model-diff-replication
    """
    This defines a data buffer, to store a stack of acts across both model that
    can be used to train the crosscoder. It'll automatically run the model to 
    generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_list, all_tokens, estimated_act_norm_list=None, do_shuffle=True):
        all_dims = [model.config.hidden_size for model in model_list]
        assert len(set(all_dims)) == 1
        self.cfg = cfg
        tot_tokens_initial_estimate = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = tot_tokens_initial_estimate // ((cfg["seq_len"] - 1) * cfg["model_batch_size"])
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        self.buffer = torch.zeros(
            (self.buffer_size, self.cfg["n_models"], self.cfg["d_in"]),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])
        self.cfg = cfg
        self.model_list = model_list
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        if "remove_bos" in cfg:
            self.remove_bos = cfg["remove_bos"]
        else:
            self.remove_bos = True
        
        if estimated_act_norm_list is None:
            estimated_act_norm_list = []
            for model in self.model_list:
                estimated_act_norm = self.estimate_act_norm(
                    cfg["model_batch_size"], 
                    model                
                )
                estimated_act_norm_list.append(estimated_act_norm)
        self.cfg["estimated_act_norm_list"] = estimated_act_norm_list
        
        self.normalisation_factor = torch.tensor(
            estimated_act_norm_list,
            device="cuda:0",
            dtype=torch.float32,
        )
        self.do_shuffle = do_shuffle
        print("Normalization factors are: ", self.normalisation_factor)
        self.refresh()

    @torch.no_grad()
    def estimate_act_norm(self, batch_size, model, n_batches_for_norm_estimate: int = int(1e3)):
        # Code mostly from: https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            with model.trace(tokens, **tracer_kwargs):
                LAYER = int(self.cfg["hook_point"].split(".")[1]) - 1
                if "gemma" in self.cfg["model_name"].lower():
                    submodule = model.model.layers[LAYER]
                elif "pythia" in self.cfg["model_name"].lower():
                    submodule = model.gpt_neox.layers[LAYER]
                elif "olmo" in self.cfg["model_name"].lower():
                    submodule = model.model.layers[LAYER]
                elif "bloom" in self.cfg["model_name"].lower():
                    submodule = model.transformer.h[LAYER]
                else:
                    raise NotImplementedError("Model name not supported yet.")

                # Capture hidden states (activations)
                hidden_states = submodule.output.save()
                curr_input = model.inputs.save()

                # Stop capturing after saving
                submodule.output.stop()
            
            # Extract activations and attention mask
            attn_mask = curr_input.value[1]["attention_mask"]
            hidden_states = hidden_states.value

            # Handle tuple outputs (e.g., transformer outputs)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # Apply attention mask (keep only non-padding tokens)
            hidden_states = hidden_states[attn_mask != 0]

            # Compute mean norm per batch
            norms_per_batch.append(hidden_states.norm(dim=-1).mean().item())
        
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.cfg["d_in"]) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        print("Refreshing the buffer!")
        self.pointer = 0
        self.buffer = torch.zeros(
            (self.buffer_size, self.cfg["n_models"], self.cfg["d_in"]),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(self.cfg["device"])
        torch.cuda.empty_cache()
        self.first = False
        with torch.autocast("cuda", torch.bfloat16):
            for _ in tqdm.trange(0, self.buffer_batches):
                tokens = self.all_tokens[self.token_pointer : self.token_pointer + self.cfg["model_batch_size"]]
                # assert tokens.shape == (self.cfg["model_batch_size"], self.cfg["seq_len"])
                cache_list = []
                for model_A in self.model_list:
                    with model_A.trace(
                        tokens,
                        **tracer_kwargs,
                    ):
                        LAYER = int(self.cfg["hook_point"].split(".")[1]) - 1
                        if "gemma" in self.cfg["model_name"].lower():
                            submodule = model_A.model.layers[LAYER]
                        elif "pythia" in self.cfg["model_name"].lower():
                            submodule = model_A.gpt_neox.layers[LAYER]
                        elif "olmo" in self.cfg["model_name"].lower():
                            submodule = model_A.model.layers[LAYER]
                        elif "bloom" in self.cfg["model_name"].lower():
                            submodule = model_A.transformer.h[LAYER]
                        else:
                            raise NotImplementedError("Model name not supported yet.")
                        hidden_states = submodule.output.save()
                        curr_input = model_A.inputs.save()
                        submodule.output.stop()
                    
                    attn_mask = curr_input.value[1]["attention_mask"]
                    hidden_states = hidden_states.value
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    hidden_states = hidden_states[attn_mask != 0]
                    hidden_states = hidden_states.view(tokens.shape[0], tokens.shape[1], hidden_states.shape[-1])
                    cache_list.append(hidden_states)

                acts = torch.stack([cache_A for cache_A in cache_list], dim=0)
                if self.remove_bos:
                    acts = acts[:, :, 1:, :] # Drop BOS
                # [n_models, batch, seq_len, d_model]
                assert acts.shape == (self.cfg["n_models"], tokens.shape[0], tokens.shape[1]-1, self.cfg["d_in"])
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )
                
                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                
        # Check that the buffer is filled
        assert self.buffer.shape[0] == self.pointer, f"Buffer size {self.buffer.shape} does not match pointer {self.pointer}"
        assert torch.sum(self.buffer[-1].abs()) > 1, f"Last batch in buffer is not refreshed \n\n {self.buffer[-1]}"

        self.pointer = 0
        if self.do_shuffle:
            self.buffer = self.buffer[
                torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
            ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer + self.cfg["batch_size"] > self.buffer.shape[0]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
    
    @torch.no_grad()
    def restart(self):
        # NOTE: this is fine as long as your batch size * batch num is less than the buffer size
        #       otherwise, if it starts refreshing, setting the pointer to 0 doesn't do much,
        #       you would need to also maybe set the self.token_pointer to 0.
        self.pointer = 0