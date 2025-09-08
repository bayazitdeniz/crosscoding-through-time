from utils import *
from trainer import TrainerNNSight, BatchTopKTrainerNNSight


def get_args(description='arguments for train_crosscoder.py main func', jupyter=False):
    parser = argparse.ArgumentParser(description=description)
    ############################################################################
    # General CrossCoder Args
    ############################################################################
    parser.add_argument(
        "--model_name",
        type=str, 
        help="e.g. 'EleutherAI/pythia-1b' or 'allenai/OLMo-1B-hf' or 'bigscience/bloom-1b1-intermediate'.")
    parser.add_argument(
        "--wandb_name",
        type=str, 
        help="e.g. whatever you want it to be ;)")
    parser.add_argument(
        "--revision_list",
        type=str, 
        help="e.g. for pythia 'step1-step2-step1000-main', for olmo 'step1000-tokens4B-step102000-tokens427B'.")
    parser.add_argument(
        "--dict_size",
        type=int,
        help="e.g. 2**14")
    parser.add_argument(
        "--seed",
        type=int,
        default=4352,
        help="e.g. 4352, random seed for the experiments")
    parser.add_argument(
        "--l1_coeff",
        type=float,
        default=2,
        help="e.g. 2, l1 regularization coefficient")
    parser.add_argument(
        "--l1_warmup_pct",
        type=float,
        default=0.05,
        help="e.g. 0.05, l1 warmup percentage")
    parser.add_argument(
        "--hook_point",
        type=str,
        default="blocks.10.hook_resid_pre",
        help="e.g. blocks.10.hook_resid_pre")
    parser.add_argument(
        "--model_batch_size",
        type=int,
        default=4,
        help="e.g. 1, 2, 4 ...")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="e.g. 4096, 2048, 1024 ...")
    parser.add_argument(
        "--dec_init_norm",
        type=float,
        default=0.08,
        help="e.g. 0.08, 1.0")
    ############################################################################
    # BatchTopKCrossCoder-Specific Args
    ############################################################################
    parser.add_argument(
        "--batch_topk_final",
        type=int,
        default=None,
        help="e.g. 200")
    parser.add_argument(
        "--batch_topk_init",
        type=int,
        default=1000,
        help="e.g. 1000")
    parser.add_argument(
        "--topk_annealing_steps",
        type=int,
        default=5000,
        help="e.g. 5000")
    parser.add_argument(
        "--threshold_start_step",
        type=int,
        default=1000,
        help="e.g. 1000")
    #### the following is fixed to be a certain default
    parser.add_argument(
        "--threshold_beta",
        type=float,
        default=0.999,
        help="e.g. 0.999")
    parser.add_argument(
        "--dead_feature_threshold",
        type=int,
        default=10_000_000,
        help="e.g. 10_000_000") 
    parser.add_argument(
        "--auxk_alpha",
        type=float,
        default=1.0/32.0,
        help="e.g. 0.03125")
    args = None

    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    
    if "olmo" in args.model_name.lower():
        args.revision_list = args.revision_list.split("_")
        for rev in args.revision_list:
            if "-" not in rev and rev != "main":
                raise ValueError("Revision list must contain '-' for olmo models or must be the 'main' checkpoint.")
    else:
        args.revision_list = args.revision_list.split("-")

    return args


def main():
    args = get_args()
    set_seed(args.seed)
    device = 'cuda:0'

    print("Revision list given is: ", args.revision_list)

    model_list = load_revision_nnsight(
        model_name=args.model_name,
        revision_list=args.revision_list,
        device=device,
        seed=args.seed
    )

    train_tokens = None 
    val_tokens = None

    args.do_ce_eval = False
    args.nnsight = True
    args.model_type = "-".join([f"{args.model_name}/{revision}" for revision in args.revision_list])
    args.n_models = len(model_list)

    if "pythia" in args.model_name.lower():
        train_tokens = load_pile_pythia_filtered(is_val=False)
        val_tokens = load_pile_pythia_filtered(is_val=True)
    elif "olmo" in args.model_name.lower():
        train_tokens = load_olmo_dolma_filtered(is_val=False)
        val_tokens = load_olmo_dolma_filtered(is_val=True)
    elif "bloom" in args.model_name.lower():
        train_tokens = load_bloom_c4(is_val=False)
        val_tokens = load_bloom_c4(is_val=True)
    else:
        raise NotImplementedError("Model name not supported yet.")

    print("Full train dataset size: ", len(train_tokens))
    print("Full val dataset size: ", len(val_tokens))
    print("First model's config: ", model_list[0].config)

    default_cfg = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "buffer_mult": 128,
        "num_val_batches": 30,
        "lr": 5e-5,
        "num_tokens": 400_000_000,
        "l1_warmup_pct":args.l1_warmup_pct,
        "l1_coeff": args.l1_coeff,
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in": model_list[0].config.hidden_size,
        "dict_size": args.dict_size,
        "seq_len": 1024,
        "enc_dtype": "fp32",
        "model_name": args.model_name,
        "revision_list": "_".join(args.revision_list) if "olmo" in args.model_name.lower() else "-".join(args.revision_list),
        "site": "resid_pre",
        "device": device,
        "model_batch_size": args.model_batch_size,
        "log_every": 1,
        "val_log_every": 100,
        "save_every": 5000,
        "dec_init_norm": args.dec_init_norm,
        "hook_point": args.hook_point,
        "wandb_project": "crosscoding-through-time",
        "wandb_entity": "your-wandb-entity",
        "wandb_name": args.wandb_name,
        "n_models": args.n_models,
        "model_type": args.model_type,
        "do_ce_eval": args.do_ce_eval,
        "nnsight": args.nnsight,
        "remove_bos": True,
        "estimated_act_norm_list": None,
        ############################################################################
        "batch_topk_final": args.batch_topk_final,
        "batch_topk_init": args.batch_topk_init,
        "topk_annealing_steps": args.topk_annealing_steps,
        "threshold_start_step": args.threshold_start_step,
        "threshold_beta": args.threshold_beta,
        "dead_feature_threshold": args.dead_feature_threshold,
        "auxk_alpha": args.auxk_alpha
    }
    cfg = arg_parse_update_cfg(default_cfg)

    set_seed(args.seed)

    if args.batch_topk_final is None:
        print("Training with L1 sparsity loss.")
        trainer = TrainerNNSight(
            cfg=cfg, 
            model_list=model_list, 
            train_tokens=train_tokens,
            val_tokens=val_tokens
        )
    else:
        print("Training with BatchTopK loss.")
        trainer = BatchTopKTrainerNNSight(
            cfg=cfg,
            model_list=model_list,
            train_tokens=train_tokens,
            val_tokens=val_tokens
        )
    print("Final config: ", trainer.crosscoder.cfg)
    print("Normalization factors are: ", trainer.crosscoder.cfg["estimated_act_norm_list"])
    trainer.train(do_ce_eval=args.do_ce_eval)


if __name__ == "__main__":
    main()
