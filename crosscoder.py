
from utils import *

from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class CrossCoder(nn.Module):
    # Code initially from: https://github.com/ckkissane/crosscoder-model-diff-replication
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        n_models = self.cfg["n_models"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        
        self.W_enc = nn.Parameter(
            torch.empty(n_models, d_in, d_hidden, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, n_models, d_in, dtype=self.dtype
                )
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((n_models, d_in), dtype=self.dtype)
        )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            f = F.relu(x_enc + self.b_enc)
        else:
            f = x_enc + self.b_enc
        return f

    def decode(self, f, apply_bias=True):
        # f: [batch, d_hidden]
        f_dec = einops.einsum(
            f,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        if apply_bias:
            return f_dec + self.b_dec
        else: 
            return f_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        f = self.encode(x)
        return self.decode(f)
    
    def create_save_dir(self):
        # TODO: fix base dir naming to be model agnostic
        #       for now keep it this way because 
        #       the path is hardcoded in the analysis
        base_dir = Path("./workspace/logs/checkpoints")
        version_list = [
            int(file.name.split("_")[1])
            for file in list(base_dir.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f, indent=4)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
        path: str = "blocks.14.hook_resid_pre",
        device: Optional[Union[str, torch.device]] = None
    ) -> "CrossCoder":
        """
        Load CrossCoder weights and config from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)
            
        Returns:
            Initialized CrossCoder instance
        """

        # Download config and weights
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cfg.json"
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cc_weights.pt"
        )

        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict)

        return instance

    @classmethod
    def load(cls, version_dir, checkpoint_version, path="./workspace/logs/checkpoints", verbose=True):
        # TODO: fix base dir naming to be model agnostic
        #       for now keep it this way because 
        #       the path is hardcoded in the analysis
        save_dir = Path(path) / str(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        if verbose:
            pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(weight_path))
        return self


class BatchTopKCrossCoder(CrossCoder):
    # Code mostly from: https://github.com/science-of-finetuning/dictionary_learning/blob/main/dictionary_learning/dictionary.py
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.k setup with register_buffer
        self.register_buffer("k", torch.tensor(cfg["batch_topk_init"], dtype=torch.int))
        threshold = -1.0
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def encode(self, x, apply_relu=True, return_all=False):        
        # x: [batch, n_models, d_model]
        batch_size = x.size(0)
        post_relu_f = super().encode(x, apply_relu=apply_relu)
        code_normalization = self.W_dec.norm(dim=2).sum(dim=1).unsqueeze(0)
        post_relu_f_scaled = post_relu_f * code_normalization
        
        f = None
        if self.training:
            # print("encode in training mode")
            # Flatten and perform batch top-k
            flattened_acts_scaled = post_relu_f_scaled.flatten()
            post_topk = flattened_acts_scaled.topk(
                self.k * batch_size, sorted=False, dim=-1
            )
            post_topk_values = post_relu_f.flatten()[post_topk.indices]
            f = (
                torch.zeros_like(flattened_acts_scaled)
                .scatter_(-1, post_topk.indices, post_topk_values)
                .reshape(post_relu_f.shape)
            )
        else:
            # print("encode in eval mode")
            f = post_relu_f * (post_relu_f_scaled > self.threshold)
        
        if return_all:
            return (
                f,
                f * code_normalization,
                f.sum(0) > 0,
                post_relu_f,
                post_relu_f_scaled,
            )
        else:
            return f
