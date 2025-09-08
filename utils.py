import os
import random
import glob
from transformers import AutoTokenizer
from transformers import set_seed as transformers_set_seed
from nnsight import LanguageModel

import plotly.io as pio
pio.renderers.default = "jupyterlab"

import einops
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import wandb

from collections import defaultdict
from datasets import (
    load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
)

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    
from constants import scaling_factor_dict

DEBUG=False
tracer_kwargs = {}
if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}

################################################################################
# GENERAL UTILS
################################################################################

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers_set_seed(seed)

def str2bool(v):
    # NOTE:
    # taken from
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parse_update_cfg(default_cfg):
    if get_ipython() is not None:
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg

def avg_std_dictlist(metric_list):
    all_keys = metric_list[0].keys()
    values_dict = {key: [] for key in all_keys}

    for d in metric_list:
        for key in all_keys:
            values_dict[key].append(d[key])

    stats_dict = {}
    for key, value_list in values_dict.items():
        arr = np.array(value_list)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        stats_dict.update({
            f'{key}-mean': mean_val,
            f'{key}-std': std_val
        })
        
    return stats_dict

################################################################################
# MODEL UTILS
################################################################################

def load_revision_nnsight(
        model_name,
        revision_list,
        device,
        seed,
        dtype=None
    ):  
    set_seed(seed)
    model_list = []
    tokenizer = None
    if "bloom" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
    for revision in revision_list:
        print(f"Loading Model Name: {model_name}, Revision: {revision}")
        model = LanguageModel(model_name, revision=revision, device_map=device, tokenizer=tokenizer)
        model.eval()
        if dtype is not None:
            model = model.to(dtype)
        model_list.append(model)
    return model_list

def get_submodule_at_layernum(model, model_name, layer_num):
    lowered_model_name = model_name.lower()
    submodule = None
    if "gemma" in lowered_model_name:
        submodule = model.model.layers[layer_num]
    elif "pythia" in lowered_model_name:
        submodule = model.gpt_neox.layers[layer_num]
    elif "olmo" in lowered_model_name:
        submodule = model.model.layers[layer_num]
    elif "bloom" in lowered_model_name:
        submodule = model.transformer.h[layer_num]
    else:
        raise NotImplementedError("Model name not supported yet.")
    
    return submodule

################################################################################
# LMODELING DATASET UTILS
################################################################################

def load_or_create_tensor_data(pt_path, hf_path, split="train", cache_columns=["input_ids"]):
    try:
        print(f"Loading data from disk: {pt_path}")
        return torch.load(pt_path, weights_only=True)
    except Exception as e:
        print(f"Could not load from {pt_path}. Reason: {e}")
        print(f"Loading from HF dataset at {hf_path}, split={split}")
        data = load_from_disk(hf_path)
        data_split = data[split]
        data_split.set_format(type="torch", columns=cache_columns)
        all_tokens = data_split["input_ids"]
        torch.save(all_tokens, pt_path)
        print(f"Saved tokens to {pt_path}")
        return all_tokens

def load_bloom_c4_uniform_multiblimp():
    """Uniform according to languages overlapping in BLOOM top-10 and MultiBLiMP."""
    return load_or_create_tensor_data(
        pt_path=os.path.join(".", "workspace", "tensor_data", "bloom_c4_train_uniform_multiblimp.pt"),
        hf_path=os.path.join(".", "workspace", "c4cache", "c4_bloom_multilingual_split_uniform_multiblimp.hf")
    )

def load_bloom_c4_uniform():
    """Uniform according to top-5 in BLOOM."""
    return load_or_create_tensor_data(
        pt_path=os.path.join(".", "workspace", "tensor_data", "bloom_c4_train_uniform.pt"),
        hf_path=os.path.join(".", "workspace", "c4cache", "c4_bloom_multilingual_split_uniform.hf"),
        split="train"
    )

def load_bloom_c4(is_val=False):
    """Load all top10 in BLOOM."""
    pt_split = "val" if is_val else "train"
    hf_split = "validation" if is_val else "train"
    pt_path = os.path.join(".", "workspace", "tensor_data", f"bloom_c4_{pt_split}.pt")
    return load_or_create_tensor_data(
        pt_path=pt_path,
        hf_path=os.path.join(".", "workspace", "c4cache", "c4_bloom_multilingual_split.hf"),
        split=hf_split
    )
    
def load_olmo_dolma_filtered(is_val=False):
    """Load OLMo Dolma filtered dataset."""
    pt_split = "val" if is_val else "train"
    hf_split = "validation" if is_val else "train"
    pt_path = os.path.join(".", "workspace", "tensor_data", f"dolma_tokenized_filtered_{pt_split}.pt")
    return load_or_create_tensor_data(
        pt_path=pt_path,
        hf_path=os.path.join(".", "workspace", "cache", "dolma_olmo_tokenized_filtered_split.hf"),
        split=hf_split
    )

def load_pile_pythia_filtered(is_val=False):
    """Load filtered Pile dataset for Pythia."""
    pt_split = "val" if is_val else "train"
    hf_split = "validation" if is_val else "train"
    pt_path = os.path.join(".", "workspace", "tensor_data", f"pile_pythia_tokenized_filtered_{pt_split}.pt")
    return load_or_create_tensor_data(
        pt_path=pt_path,
        hf_path=os.path.join(".", "workspace", "cache", "pile_pythia_tokenized_filtered.hf"),
        split=hf_split
    )

################################################################################
# SV-AGREEMENT DATASET UTILS
################################################################################

def load_blimp_subject_verb(tokenizer=None, task=None, is_token_version=True):
    """Load and process BLiMP subject-verb datasets."""
    
    def load_jsonl(path):
        with open(path, "r") as f:
            return [json.loads(line) for line in f]
        
    def _process_doc_one_prefix(doc):
        out_doc = {}
        
        if is_token_version:
            # find common prefix through tokenization 
            # (in this dataset it doesn't make a difference as it's already broken by word)
            ta = tokenizer.encode(doc["one_prefix_prefix"].strip() + " " + doc["one_prefix_word_good"].strip(), add_special_tokens=False)
            tb = tokenizer.encode(doc["one_prefix_prefix"].strip() + " " + doc["one_prefix_word_bad"].strip(), add_special_tokens=False)
            
            common_len = 0
            for x, y in zip(ta, tb):
                if x != y:
                    break
                common_len += 1
            
            out_doc["clean_prefix"] = tokenizer.decode(ta[:common_len], clean_up_tokenization_spaces=False)
            out_doc["clean_answer"] = tokenizer.decode(ta[common_len:], clean_up_tokenization_spaces=False)
            out_doc["patch_answer"] = tokenizer.decode(tb[common_len:], clean_up_tokenization_spaces=False)
        else:
            out_doc["clean_prefix"] = doc["one_prefix_prefix"].strip()
            out_doc["clean_answer"] = " " + doc["one_prefix_word_good"].strip()
            out_doc["patch_answer"] = " " + doc["one_prefix_word_bad"].strip()
        out_doc["skip"] = False
        
        if tokenizer is not None:
            clean_tokens = tokenizer(out_doc["clean_answer"])["input_ids"]
            patch_tokens = tokenizer(out_doc["patch_answer"])["input_ids"]
            if (len(clean_tokens) != len(patch_tokens)) or (len(clean_tokens) > 2):
                # skip the case where they are not equal length 
                # or splits to more than two tokens, as it's complicated to handle
                out_doc["skip"] = True
                return out_doc
            
            # (1) if length is same and == 1 no need to change clean and patch answer
            if len(clean_tokens) > 1:
                
                # (2) if length is same and == 2, and first token makes the diff, 
                # put the first tokens as the answers
                clean_main = clean_tokens[0]
                patch_main = patch_tokens[0]
                out_doc["clean_answer"] = tokenizer.decode(clean_tokens[0])
                out_doc["patch_answer"] = tokenizer.decode(patch_tokens[0])
                
                # (3) otherwise the second token must be making the difference, 
                # add that to answers, and the first token to the prefix
                if clean_main == patch_main:
                    out_doc["clean_prefix"] = out_doc["clean_prefix"] + tokenizer.decode(clean_tokens[0])
                    out_doc["clean_answer"] = tokenizer.decode(clean_tokens[1])
                    out_doc["patch_answer"] = tokenizer.decode(patch_tokens[1])
                    
        return out_doc
    
    processed = []
    task_files = []
    if task is None:
        task_files = [
            "distractor_agreement_relational_noun.jsonl", 
            "distractor_agreement_relative_clause.jsonl", 
            "irregular_plural_subject_verb_agreement_1.jsonl", 
            "irregular_plural_subject_verb_agreement_2.jsonl", 
            "regular_plural_subject_verb_agreement_1.jsonl", 
            "regular_plural_subject_verb_agreement_2.jsonl" 
        ]
    else:
        task_files = [f"{task}.jsonl"]
        
    task_paths = [os.path.join("data", "blimp_data", task_file) for task_file in task_files]
    
    for task_path in task_paths:
        # read the file and separate one_prefix and two_prefix datapoints
        task_data = load_jsonl(task_path)
        is_one_prefix = all([d["one_prefix_method"] for d in task_data])
        is_two_prefix = all([d["two_prefix_method"] for d in task_data])
        
        # if all one_prefix datapoints, process to get clean and patch answers
        if is_one_prefix and not is_two_prefix:
            print("-" * 10)
            print(task_path)
            dataset = Dataset.from_list(task_data)
            print(len(dataset))
            dataset = dataset.map(_process_doc_one_prefix)
            dataset = dataset.filter(lambda x: not x["skip"])
            print(len(dataset))
            processed.append(dataset)
        # if all two_prefix then skip this task
        elif is_two_prefix and not is_one_prefix:
            continue
        # otherwise, this line should never be reached, keep this error in case
        else:
            raise ValueError("Should be either all two prefix or all one prefix, but that isn't the case.")

    combined_dataset = concatenate_datasets(processed)
    return DatasetDict({"train": combined_dataset})

def load_clams_helper(tokenizer, lang, task):
    
    def load_clams_file(path_to_txt):
        df_raw = pd.read_csv(path_to_txt, sep="\t", header=None, names=['is_true', 'sentence'])

        # split into correct/wrong pairs with 'is_true' flag
        trues = df_raw[df_raw['is_true'] == True].reset_index(drop=True)
        falses = df_raw[df_raw['is_true'] == False].reset_index(drop=True)

        # Build structured DataFrame
        records = []
        for t, f in zip(trues['sentence'], falses['sentence']):
            # common prefix words
            t, f = t.capitalize(), f.capitalize()
            wt, wf = t.split(), f.split()
            prefix_words = []
            for w1, w2 in zip(wt, wf):
                if w1 == w2:
                    prefix_words.append(w1)
                else:
                    break
            prefix = ' '.join(prefix_words)
            clean_ans = ' ' + wt[len(prefix_words)]
            patch_ans = ' ' + wf[len(prefix_words)]
            records.append({
                "sentence_good": t,
                "sentence_bad": f,
                'clean_prefix': prefix,
                'clean_answer': clean_ans,
                'patch_answer': patch_ans,
                "skip": False
            })
        return Dataset.from_list(records)

    
    def _process_doc_clams(out_doc):
        clean_tokens = tokenizer(out_doc["clean_answer"])["input_ids"]
        patch_tokens = tokenizer(out_doc["patch_answer"])["input_ids"]
        # Skip the case where they are not equal length or more than equal to 3 tokens
        if (len(clean_tokens) != len(patch_tokens)) or (len(clean_tokens) > 2):
            out_doc["skip"] = True
            return out_doc
        
        if len(clean_tokens) > 1:
            clean_main = clean_tokens[0]
            patch_main = patch_tokens[0]
            out_doc["clean_answer"] = tokenizer.decode(clean_tokens[0])
            out_doc["patch_answer"] = tokenizer.decode(patch_tokens[0])
            
            if clean_main == patch_main:
                out_doc["clean_prefix"] = out_doc["clean_prefix"] + tokenizer.decode(clean_tokens[0])
                out_doc["clean_answer"] = tokenizer.decode(clean_tokens[1])
                out_doc["patch_answer"] = tokenizer.decode(patch_tokens[1])
        return out_doc    
    
    dataset_list = []

    dataset = load_clams_file(f"../clams/{lang}_evalset/{task}.txt")
    dataset = dataset.map(lambda x: _process_doc_clams(x))
    dataset = dataset.filter(lambda x: not x["skip"])
    
    dataset.to_json(os.path.join("data", "clams_data", f"{lang}_{task}.jsonl"), lines=True, force_ascii=False)
    
    print(len(dataset))
    
    return DatasetDict({
        "train": dataset
    })

def load_clams_helper_token(tokenizer, lang, task):
    """
    Load CLAMS data using token-level prefix detection in a two-stage map/filter pipeline,
    ensuring all mapped docs define the same schema fields.
    """
    def load_clams_file(path_to_txt):
        df = pd.read_csv(path_to_txt, sep="\t", header=None,
                         names=["is_true", "sentence"], dtype={"is_true": bool, "sentence": str})
        trues = df[df["is_true"]].reset_index(drop=True)
        falses = df[~df["is_true"]].reset_index(drop=True)

        records = []
        for true_sent, false_sent in zip(trues["sentence"], falses["sentence"]):
            t = true_sent.strip().capitalize()
            f = false_sent.strip().capitalize()
            records.append({
                "sentence_good": t,
                "sentence_bad": f,
                "skip": False
            })
        return Dataset.from_list(records)

    def _process_doc_token(out_doc):
        # initialize all expected fields for consistent schema
        out_doc.setdefault("clean_prefix", "")
        out_doc.setdefault("clean_answer", "")
        out_doc.setdefault("patch_answer", "")
        out_doc.setdefault("skip", False)

        # Tokenize full sentences without special tokens
        t_ids = tokenizer.encode(out_doc["sentence_good"], add_special_tokens=False)
        f_ids = tokenizer.encode(out_doc["sentence_bad"], add_special_tokens=False)

        # Find longest common token prefix
        common_len = 0
        for id_t, id_f in zip(t_ids, f_ids):
            if id_t == id_f:
                common_len += 1
            else:
                break
        prefix_ids = t_ids[:common_len]
        clean_ids = t_ids[common_len:]
        patch_ids = f_ids[common_len:]

        # Skip mismatched-length answers or >2 tokens
        if len(clean_ids) != len(patch_ids) or len(clean_ids) > 2:
            out_doc["skip"] = True
            return out_doc

        # Handle two-token answers with shared first token
        if len(clean_ids) == 2 and clean_ids[0] == patch_ids[0]:
            prefix_ids.append(clean_ids[0])
            clean_ids = clean_ids[1]
            patch_ids = patch_ids[1]
        elif len(clean_ids) == 2 and clean_ids[0] != patch_ids[0]:
            clean_ids = clean_ids[0]
            patch_ids = patch_ids[0]

        # Decode segments
        out_doc["clean_prefix"] = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False)
        out_doc["clean_answer"] = tokenizer.decode(clean_ids, clean_up_tokenization_spaces=False)
        out_doc["patch_answer"] = tokenizer.decode(patch_ids, clean_up_tokenization_spaces=False)
        return out_doc

    # Paths and loading
    input_path = os.path.join("..", "clams", f"{lang}_evalset", f"{task}.txt")
    ds = load_clams_file(input_path)

    # Token-level prefix split
    ds = ds.map(_process_doc_token)
    # Filter out skipped items
    ds = ds.filter(lambda x: not x["skip"])

    # Save
    out_dir = os.path.join("data", "clams_data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{lang}_{task}_tokenprefix.jsonl")
    ds.to_json(out_file, lines=True, force_ascii=False)
    print(f"Processed {len(ds)} examples, saved to {out_file}.")

    return DatasetDict({"train": ds})

def load_clams(
        tokenizer, 
        langs=["en", "fr"], 
        tasks=["long_vp_coord", "obj_rel_across_anim", "obj_rel_within_anim", 
                "prep_anim", "simple_agrmt", "subj_rel", "vp_coord"],
        is_token_version=False
    ):
    """
    Load and process CLAMS subject-verb datasets.

    For string prefix finding:
    og_task_lang_sizes = {
        'long_vp_coord-en': 320, 
        'long_vp_coord-fr': 300, 
        'obj_rel_across_anim-en': 6400, 
        'obj_rel_across_anim-fr': 9600, 
        'obj_rel_within_anim-en': 5600, 
        'obj_rel_within_anim-fr': 5600, 
        'prep_anim-en': 9600, 
        'prep_anim-fr': 12000, 
        'simple_agrmt-en': 80, 
        'simple_agrmt-fr': 240, 
        'subj_rel-en': 6400, 
        'subj_rel-fr': 9600, 
        'vp_coord-en': 480, 
        'vp_coord-fr': 840}
        
    For token prefix finding:
    """
    task_lang_sizes = {
        'long_vp_coord-en': 300, 
        'long_vp_coord-fr': 300, 
        'obj_rel_across_anim-en': 400, 
        'obj_rel_across_anim-fr': 400, 
        'obj_rel_within_anim-en': 400, 
        'obj_rel_within_anim-fr': 400, 
        'prep_anim-en': 400, 
        'prep_anim-fr': 400, 
        'simple_agrmt-en': 80, 
        'simple_agrmt-fr': 80, 
        'subj_rel-en': 400, 
        'subj_rel-fr': 400, 
        'vp_coord-en': 400, 
        'vp_coord-fr': 400
    }
    
    dsets = []

    for task in tasks:
        for lang in langs:
            key = f"{task}-{lang}"
            num_samples = task_lang_sizes.get(key, 0)  # default to 0 if missing
            if is_token_version:
                dataset = load_clams_helper_token(tokenizer=tokenizer, lang=lang, task=task)["train"].shuffle(seed=42)
            else:
                dataset = load_clams_helper(tokenizer=tokenizer, lang=lang, task=task)["train"].shuffle(seed=42)
            dsets.append(dataset.select(range(min(num_samples, len(dataset)))))
            # dsets.append(dataset)

    dataset = DatasetDict({
        "train": concatenate_datasets(dsets).shuffle(seed=42)
    })

    return dataset

def load_multiblimp(
        tokenizer, 
        langs=["eng", "fra", "spa", "por", "arb", "hin"], # "ben"], 
        tasks=["SV-#", "SV-G", "SV-P"],
        uniform_sample=True,
        is_token_version=False
    ):
    
    def _process_doc_multiblimp(doc, lang):
        t = doc['sen']
        f = doc['wrong_sen']
        wt, wf = t.split(), f.split()
        prefix_words = []
        for w1, w2 in zip(wt, wf):
            if w1 == w2:
                prefix_words.append(w1)
            else:
                break
        prefix = ' '.join(prefix_words)
        clean_ans = ' ' + wt[len(prefix_words)]
        patch_ans = ' ' + wf[len(prefix_words)]
        
        out_doc = {
            "sentence_good": t,
            "sentence_bad": f,
            'clean_prefix': prefix,
            'clean_answer': clean_ans,
            'patch_answer': patch_ans,
            "skip": False,
            "lang": lang
        }
        
        clean_tokens = tokenizer(out_doc["clean_answer"])["input_ids"]
        patch_tokens = tokenizer(out_doc["patch_answer"])["input_ids"]
        # Skip the case where they are not equal length or more than equal to 3 tokens
        if (len(clean_tokens) != len(patch_tokens)) or (len(clean_tokens) > 2):
            out_doc["skip"] = True
            return out_doc
        
        if len(clean_tokens) > 1:
            clean_main = clean_tokens[0]
            patch_main = patch_tokens[0]
            out_doc["clean_answer"] = tokenizer.decode(clean_tokens[0])
            out_doc["patch_answer"] = tokenizer.decode(patch_tokens[0])
            
            if clean_main == patch_main:
                out_doc["clean_prefix"] = out_doc["clean_prefix"] + tokenizer.decode(clean_tokens[0])
                out_doc["clean_answer"] = tokenizer.decode(clean_tokens[1])
                out_doc["patch_answer"] = tokenizer.decode(patch_tokens[1])
        
        return out_doc

    def _process_doc_multiblimp_token(doc, lang):
        out_doc = {}
        out_doc.setdefault("clean_prefix", "")
        out_doc.setdefault("clean_answer", "")
        out_doc.setdefault("patch_answer", "")
        out_doc.setdefault("skip", False)
        
        t = doc['sen']
        f = doc['wrong_sen']
        out_doc["sentence_good"] = t
        out_doc["sentence_bad"] = f

        # Tokenize full sentences without special tokens
        t_ids = tokenizer.encode(t, add_special_tokens=False)
        f_ids = tokenizer.encode(f, add_special_tokens=False)

        # Find longest common token prefix
        common_len = 0
        for id_t, id_f in zip(t_ids, f_ids):
            if id_t == id_f:
                common_len += 1
            else:
                break
        prefix_ids = t_ids[:common_len]
        clean_ids = t_ids[common_len:]
        patch_ids = f_ids[common_len:]

        # Skip mismatched-length answers or >2 tokens
        if len(clean_ids) != len(patch_ids) or len(clean_ids) > 2:
            out_doc["skip"] = True
            return out_doc

        # Handle two-token answers with shared first token
        if len(clean_ids) == 2 and clean_ids[0] == patch_ids[0]:
            prefix_ids.append(clean_ids[0])
            clean_ids = clean_ids[1]
            patch_ids = patch_ids[1]
        elif len(clean_ids) == 2 and clean_ids[0] != patch_ids[0]:
            clean_ids = clean_ids[0]
            patch_ids = patch_ids[0]

        # Decode segments
        out_doc["clean_prefix"] = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False)
        out_doc["clean_answer"] = tokenizer.decode(clean_ids, clean_up_tokenization_spaces=False)
        out_doc["patch_answer"] = tokenizer.decode(patch_ids, clean_up_tokenization_spaces=False)
        
        return out_doc    
    
    task_lang_sizes = {
        'SV-#-eng': 100, 
        'SV-#-fra': 100, 
        'SV-#-spa': 100, 
        'SV-#-por': 100, 
        'SV-#-arb': 100, 
        'SV-#-hin': 100, 
        #
        'SV-G-eng': 0, 
        'SV-G-fra': 0, 
        'SV-G-spa': 0, 
        'SV-G-por': 100,
        'SV-G-arb': 100, 
        'SV-G-hin': 100,
        # 
        'SV-P-eng': 290, 
        'SV-P-fra': 290, 
        'SV-P-spa': 290, 
        'SV-P-por': 290, 
        'SV-P-arb': 290, 
        'SV-P-hin': 290
    }
    
    dsets = []
    for task in tasks:
        for lang in langs:
            key = f"{task}-{lang}"
            print("Processing: ", key)
            num_samples = task_lang_sizes.get(key, 0)  # default to 0 if missing            
            dataset = load_dataset("jumelet/multiblimp", lang, cache_dir="./workspace/cache/multiblimp/")["train"].shuffle(seed=42)
            dataset = dataset.filter(lambda doc: doc["phenomenon"] == task)
            # og_task_lang_sizes[key] = len(dataset)
            
            if is_token_version:
                dataset = dataset.map(lambda x: _process_doc_multiblimp_token(x, lang))
            else:
                dataset = dataset.map(lambda x: _process_doc_multiblimp(x, lang))
            dataset = dataset.filter(lambda x: not x["skip"])
            # task_lang_sizes[key] = len(dataset)
            
            print("-" * 10)
            print(len(dataset))
            if uniform_sample:
                dataset = dataset.select(range(min(len(dataset),num_samples)))
                
            print(len(dataset))
            dsets.append(dataset)
    
    dataset = DatasetDict({
        "train": concatenate_datasets(dsets).shuffle(seed=42)
    })
    
    return dataset

def load_task_dataset(dataset_name, tokenizer=None):
    dataset = None
    
    if dataset_name.startswith("subjectverb"):
        split_name = dataset_name.split("-")
        task = None
        if len(split_name) > 1:
            task = split_name[1]
        dataset = load_blimp_subject_verb(tokenizer=tokenizer, task=task)
    
    elif dataset_name.startswith("clams"):
        tasks = dataset_name.split("-")[1]
        langs = dataset_name.split("-")[2]
        if tasks == "all" and langs == "all":
            dataset = load_clams(tokenizer=tokenizer)
        elif tasks == "all" and langs != "all":
            dataset = load_clams(tokenizer=tokenizer, langs=[langs])
        elif tasks != "all" and langs == "all":
            dataset = load_clams(tokenizer=tokenizer, tasks=[tasks])
        else:
            dataset = load_clams(tokenizer=[langs], tasks=[tasks])
    
    elif dataset_name.startswith("multiblimp"):
        langs = dataset_name.split("_")[1]
        tasks = dataset_name.split("_")[2]
        if tasks == "all" and langs == "all":
            dataset = load_multiblimp(tokenizer=tokenizer, uniform_sample=True)
        elif tasks == "all" and langs != "all":
            dataset = load_multiblimp(tokenizer=tokenizer, langs=[langs], uniform_sample=False)
        elif tasks != "all" and langs == "all":
            dataset = load_multiblimp(tokenizer=tokenizer, tasks=[tasks], uniform_sample=True)
        else:
            dataset = load_multiblimp(tokenizer=tokenizer, langs=[langs], tasks=[tasks], uniform_sample=False)
    
    else:
        raise ValueError("Invalid dataset name.")
    
    return dataset

def print_dataset(dataset):

    for i in range(5):
        print("-" * 10)
        sample = dataset["train"][i]
        print(sample["clean_prefix"])
        print(sample["clean_answer"])
        
        if "patch_prefix" in sample:
            print(sample["patch_prefix"])
        
        if "patch_answer" in sample:
            print(sample["patch_answer"])

    print("-" * 10)
    print("All samples printed!")
