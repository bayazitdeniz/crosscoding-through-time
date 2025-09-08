import json
import math
import argparse
from ast import literal_eval
from itertools import combinations
import torch.nn.functional as F

from utils import *
from vis import revision2tokens_pythia, revision2tokens_bloom, revision2tokens_olmo

def prepare_pythia_revision_list():
    init_steps = [int(mystr) for mystr in  "0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 10000 16000".split(" ")]
    final_steps = list(range(20_000, 143_000, 10_000))
    revision_list = init_steps + final_steps + ["143000"]
    revision_list = ["step" + str(step) for step in revision_list]
    print(revision_list)
    print([revision2tokens_pythia(rev) for rev in revision_list])
    print(len(revision_list))
    return revision_list

def prepare_olmo_revision_list():
    init_steps = [int(mystr) for mystr in ["0", "1000", "2000", "4000", "8000", "10000", "16000"]]
    middle_steps = list(range(20_000, 143_000, 10000))
    final_steps = list(range(143_000, 1_454_000, 100_000))
    revision_list = init_steps + middle_steps +final_steps + [1454000]
    revision_list = ["step" + str(step) for step in revision_list]

    with open("scripts/olmo_1B_july_checkpoints.txt") as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
        
    step2rev = {}
    for line in lines:
        step = line.split("-")[0]
        step2rev[step] = line
        
    revision_list = [step2rev[step] for step in revision_list]
    
    print([revision2tokens_olmo(rev) for rev in revision_list])
    print(len(revision_list))
    return revision_list

def prepare_bloom_revision_list():
    steps = [1_000, 10_000, 100_000, 200_000, 300_000, 500_000, 600_000]
    revision_list = ["global_step" + str(step) for step in steps]
    print(revision_list)
    print([revision2tokens_bloom(rev) for rev in revision_list])
    print(len(revision_list))
    return revision_list


def prepare_activ_dataset(dataset_name, model_name, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_task_dataset(dataset_name, tokenizer=tokenizer)
    texts = [inp + tgt for inp, tgt in zip(dataset["train"]["clean_prefix"], dataset["train"]["clean_answer"])]
    num_examples = len(texts)
    n_batches = math.floor(num_examples / batch_size)
    batches = [
        texts[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    if len(batches) != 0:
        print(batches[0])
    
    return batches

def cosine_sim(A: torch.Tensor, B: torch.Tensor, dim=-1):
    """
    A, B: tensors of shape [..., D]
    Returns: cosine similarity per sample along dim.
    """
    return F.cosine_similarity(A, B, dim=dim)

def pearson_corr(A: torch.Tensor, B: torch.Tensor, dim=-1, eps=1e-8):
    """
    Compute Pearson correlation along the last dim.
    """
    A_mean = A.mean(dim=dim, keepdim=True)
    B_mean = B.mean(dim=dim, keepdim=True)
    A_c = A - A_mean
    B_c = B - B_mean
    num = (A_c * B_c).sum(dim=dim)
    den = torch.sqrt((A_c**2).sum(dim=dim) * (B_c**2).sum(dim=dim)).clamp(min=eps)
    return num / den

def get_activations(model, submodule, tokens):
    """
    Run inputs through model and capture the activations at the given submodule.
    """
    with model.trace(tokens, **tracer_kwargs):
        hidden_states = submodule.output.save()
        curr_input = model.inputs.save()
        submodule.output.stop()
        
    attn_mask = curr_input.value[1]["attention_mask"]
    hidden_states = hidden_states.value
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    # print(hidden_states.shape)
    # print(attn_mask.shape)
    hidden_states = hidden_states[attn_mask != 0]
    # print(hidden_states.shape)
    return hidden_states

def get_cached_batch_acts(
    model_name, 
    rev, 
    model, 
    submodule, 
    batch_inputs, 
    layer_num,
    batch_idx, 
    cache_dir="workspace/act_cache"
):
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{model_name.replace('/','_')}_{rev}_layer{layer_num}_batch{batch_idx}.pt"
    path  = os.path.join(cache_dir, fname)

    if os.path.exists(path):
        return torch.load(path)

    # get_activations returns [batch, seq_len, hidden_dim]
    acts = get_activations(model, submodule, batch_inputs)
    torch.save(acts, path)
    return acts

def compare_checkpoints(model_name, revs, batched_inputs, layer_num, skip_model_loading=False):
    results = {}  # (rev_a, rev_b) -> list of (cos, corr) per batch
    all_combo = list(combinations(revs, 2))
    print("Total number of combinations: ", len(all_combo))
    # all_combo = [all_combo[0]]
    # print("Total number of combinations: ", len(all_combo))
    
    for a, b in tqdm(all_combo, desc="Combos"):
        key = (a, b)
        results[key] = []
        
        model_list = [None, None]
        submodule1 = None
        submodule2 = None
        if not skip_model_loading:
            model_list = load_revision_nnsight(
                model_name=model_name,
                revision_list=[a, b],
                device="cuda:0",
                seed=42
            )
            submodule1 = get_submodule_at_layernum(
                model=model_list[0],
                model_name=model_name, 
                layer_num=layer_num
            )
            submodule2 = get_submodule_at_layernum(
                model=model_list[1],
                model_name=model_name, 
                layer_num=layer_num
            )


        # for idx, batch in enumerate(batched_inputs):
        for idx, batch in tqdm(enumerate(batches)):
            # print(idx)
            # print(batch)
            A = get_cached_batch_acts(model_name, a, model_list[0], submodule1, batch, layer_num, idx)
            B = get_cached_batch_acts(model_name, b, model_list[1], submodule2, batch, layer_num, idx)

            # each is [batch, dim]
            cos_batch  = cosine_sim(A, B, dim=1).mean().item()
            corr_batch = pearson_corr(A, B, dim=1).mean().item()
            results[key].append((cos_batch, corr_batch))

            # free memory
            del A, B
            
        # del model_list
        # del submodule1
        # del submodule2

    # aggregate
    summary = {}
    for key, vals in results.items():
        cossims, corrs = zip(*vals)
        summary[key] = {
            "avg_cosine":  sum(cossims) / len(cossims),
            "avg_pearson": sum(corrs) / len(corrs),
        }
        
    filename = f"workspace/act_cache/summary_{model_name.replace('/','_')}_layer{layer_num}.json"
    new_summary = {str(k): v for k, v in summary.items()}
    json_dict = json.dumps(new_summary,  ensure_ascii=False, indent=4) 
    f = open(filename, "w")
    f.write(json_dict)
    f.close()
            
    return summary

def main():
    parser = argparse.ArgumentParser(description="args for get_activations")
    parser.add_argument(
        "--model_name",
        type=str,
        default="olmo",
        help="EleutherAI/pythia-1b, allenai/OLMo-1B-0724-hf, or bigscience/bloom-1b1-intermediate")
    parser.add_argument(
        "--task",
        type=str,
        default="subjectverb",
        help="subjectverb or multiblimp_all_all")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32)
    parse.add_argument(
        "--layer_num",
        type=int,
        default=8,
        help="8 or 12"
    )
    args = parser.parse_args()        
    
    model_name_lowered = args.model_name.lower()
    batches = prepare_activ_dataset(
        dataset_name=args.task, 
        model_name=args.model_name, 
        batch_size=args.batch_size)
    revs = None
    if "pythia" in model_name_lowered:
        revs = prepare_pythia_revision_list()
    elif "olmo" in model_name_lowered:
        revs = prepare_olmo_revision_list()
    elif "bloom" in model_name_lowered:
        revs = prepare_bloom_revision_list()
    else:
        raise ValueError(f"Model {args.model_name} not handled yet.")
    
    summary = compare_checkpoints(
        model_name=model_name,
        revs=revs, 
        batched_inputs=batches, 
        layer_num=args.layer_num, 
        skip_model_loading=False)
    
    print(summary)

if __name__ == "__main__":
    main()