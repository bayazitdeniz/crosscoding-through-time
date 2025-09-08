import os
import argparse
import itertools
from tqdm import tqdm

from datasets import load_dataset, DatasetDict, DownloadConfig, Dataset
import datasets
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder, snapshot_download
from transformers import AutoTokenizer

# NOTE: The processing in this file has already been done and saved,
#       therefore you do not need to run it. However, we keep it here for those 
#       who would like to adapt the method to another architecture / pretraining 
#       dataset.

LMODELING_FOLDERS = [
    "pile_pythia_tokenized_filtered.hf",
    "dolma_olmo_tokenized_filtered_split.hf",
    "c4_bloom_multilingual_split.hf",
    "c4_bloom_multilingual_split_uniform.hf",
    "c4_bloom_multilingual_split_uniform_multiblimp.hf",
]

################################################################################
# Pythia + Pile
################################################################################
def tokenize_and_filter_pythia_pile():
    pile_train = load_dataset(
        "monology/pile-uncopyrighted", 
        split="train",
        cache_dir="./workspace/cache/",
    )
    print(len(pile_train))

    pile_val = load_dataset(
        "monology/pile-uncopyrighted",
        data_files={"validation": "val.jsonl.zst"},
        split="validation",
        cache_dir="./workspace/monologycache/",
    )
    print(len(pile_val))

    pile_test = load_dataset(
        "monology/pile-uncopyrighted",
        data_files={"test": "test.jsonl.zst"},
        split="test",
        cache_dir="./workspace/monologycache/",
    )
    print(len(pile_test))

    # Load Pythia-1B tokenizer
    MODEL_NAME = "EleutherAI/pythia-1b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    chunk_size = 1024

    # Function to tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=chunk_size)

    def filter_short_sequences(example):
        # Check if the sequence contains the padding token ID
        pad_token_id = tokenizer.pad_token_id
        return pad_token_id not in example['input_ids']

    train_tokenized_dataset = pile_train.map(
        tokenize_function,
        batched=True,
        num_proc=128,
        remove_columns=["text"],
    )
        
    val_tokenized_dataset = pile_val.map(
        tokenize_function,
        batched=True,
        num_proc=128,
        remove_columns=["text"],
    )

    test_tokenized_dataset = pile_test.map(
        tokenize_function,
        batched=True,
        num_proc=128,
        remove_columns=["text"],
    )

    hf_processed_dataset = DatasetDict({
        "train": train_tokenized_dataset,
        "validation": val_tokenized_dataset,
        "test": test_tokenized_dataset
    })

    # Save tokenized datasets locally
    save_path = "./workspace/cache/pile_pythia_tokenized.hf"
    hf_processed_dataset.save_to_disk(save_path)

    # Filter out short sequences
    for split in hf_processed_dataset.keys():
        hf_processed_dataset[split] = hf_processed_dataset[split].filter(
            filter_short_sequences,
            num_proc=128
        )
        
    for split in hf_processed_dataset.keys():
        print(f"Split {split} len: ", len(hf_processed_dataset[split]))
        
    # Save tokenized datasets locally
    save_path = "./workspace/cache/pile_pythia_tokenized_filtered.hf"
    hf_processed_dataset.save_to_disk(save_path)

    # Subsample each split to save space
    split_amount = {
        "train": 400_000,
        "validation": 18_000,
        "test": 18_000
    }

    mynewdataset = DatasetDict({
        split: hf_processed_dataset[split].shuffle(seed=42).select(range(split_amount[split]))
        for split in hf_processed_dataset.keys()
    })

    for split in mynewdataset.keys():
        print(f"Split {split} len: ", len(mynewdataset[split]))
        
    # Save tokenized datasets locally
    save_path = "./workspace/cache/pile_pythia_tokenized_filtered.hf"
    mynewdataset.save_to_disk(save_path)


################################################################################
# OLMo + Dolma
################################################################################
def tokenize_and_filter_olmo_dolma():
    #Load OLMo tokenizer
    from transformers import AutoTokenizer
    from datasets import load_dataset, Dataset, DatasetDict
    import tqdm
    import os

    file_pattern = 'v1_5r2_sample-*.json.gz'
    data_dir = "workspace/dolmacache"

    dataset = load_dataset('json', data_files=f'{data_dir}/{file_pattern}', streaming=True, split='train')


    MODEL_NAME = "allenai/OLMo-1B-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    chunk_size = 1024
    # # Function to tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=chunk_size)

    def filter_short_sequences(example):
        # Check if the sequence contains the padding token ID
        pad_token_id = tokenizer.pad_token_id
        return pad_token_id not in example['input_ids']

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        # num_proc=128,
        remove_columns=["text"],
    )

    filtered_dataset = tokenized_dataset.filter(filter_short_sequences)

    # Convert the iterable dataset to a list
    mydataset = []
    cnt = 0
    for data in filtered_dataset:
        if cnt % 1000 == 0:
            print(cnt)
        mydataset.append({"input_ids": data["input_ids"]})
        cnt += 1
        if cnt > 700_000:
            break

    # Create a Dataset object from the filtered data
    final_dataset = Dataset.from_list(mydataset)

    # Save the dataset to disk
    final_dataset.save_to_disk(f"{data_dir}/dolma_filtered_dataset.hf")

    print("✅ Successfully saved the dataset!")

    MODEL_NAME = "allenai/OLMo-1B-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data_dir = "workspace/dolmacache"
    hf_cache_dir = f"{data_dir}/dolma_filtered_dataset.hf"
    dataset = load_from_disk(hf_cache_dir)

    # Step 2: Shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    # Step 3: Split the dataset into train, validation, and test sets
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

    # Combine splits into a DatasetDict
    split_dataset = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    # Step 4: Save the splits locally
    hf_cache_dir = f"{data_dir}/dolma_filtered_dataset_split.hf"
    split_dataset.save_to_disk(hf_cache_dir)

    # Step 5: Upload the dataset back to Hugging Face Hub
    repo_name = "bayazitdeniz/dolma_olmo_tokenized_filtered_split"  # Replace with your username and desired repo name
    create_repo(repo_name, private=True)
    split_dataset.push_to_hub(repo_name, token="insert_private_token_here")
    print("Dataset successfully split and uploaded!")


################################################################################
# BLOOM + c4
################################################################################
def get_roots_lang_ratios():
    # table from ROOTS corpus paper
    mystring = """Akan                & aka       & ak          & Kwa                     & Niger-Congo    & Africa    & 70,1554        \\
    Arabic              & arb       & ar          & Semitic                 & Afro-Asiatic   & Eurasia   & 74,854,900,600   \\
    Assamese            & asm       & as          & Indic                   & Indo-European  & Eurasia   & 291,522,098     \\
    Bambara             & bam       & bm          & Western Mande           & Mande          & Africa    & 391,747        \\
    Basque              & eus       & eu          & Basque                  & Basque         & Eurasia   & 2,360,470,848    \\
    Bengali             & ben       & bn          & Indic                   & Indo-European  & Eurasia   & 18,606,823,104   \\
    Catalan             & cat       & ca          & Romance                 & Indo-European  & Eurasia   & 17,792,493,289   \\
    Chichewa           & nya       & ny          & Bantoid                 & Niger-Congo    & Africa    & 1,187,405       \\
    chiShona           & sna       & sn          & Bantoid                 & Niger-Congo    & Africa    & 6,638,639       \\
    Chitumbuka         & tum       & tum         & Bantoid                 & Niger-Congo    & Africa    & 170,360        \\
    English             & eng       & en          & Germanic                & Indo-European  & Eurasia   & 484,953,009,124  \\
    Fon                 & fon       & fon         & Kwa                     & Niger-Congo    & Africa    & 2,478,546       \\
    French              & fra       & fr          & Romance                 & Indo-European  & Eurasia   & 208,242,620,434  \\
    Gujarati            & guj       & gu          & Indic                   & Indo-European  & Eurasia   & 1,199,986,460    \\
    Hindi               & hin       & hi          & Indic                   & Indo-European  & Eurasia   & 24,622,119,985   \\
    Igbo                & ibo       & ig          & Igboid                  & Niger-Congo    & Africa    & 14078,521      \\
    Indonesian          & ind       & id          & Malayo-Sumbawan         & Austronesian   & Papunesia & 19,972,325,222   \\
    isiXhosa               & xho       & xh          & Bantoid                 & Niger-Congo    & Africa    & 14,304,074      \\
    isiZulu            & zul       & zu          & Bantoid                 & Niger-Congo    & Africa    & 8,511,561       \\
    Kannada             & kan       & kn          & Southern Dravidian      & Dravidian      & Eurasia   & 2,098,453,560    \\
    Kikuyu              & kik       & ki          & Bantoid                 & Niger-Congo    & Africa    & 359,615        \\
    Kinyarwanda         & kin       & rw          & Bantoid                 & Niger-Congo    & Africa    & 40,428,299      \\
    Kirundi             & run       & rn          & Bantoid                 & Niger-Congo    & Africa    & 3,272,550       \\
    Lingala             & lin       & ln          & Bantoid                 & Niger-Congo    & Africa    & 1,650,804       \\
    Luganda             & lug       & lg          & Bantoid                 & Niger-Congo    & Africa    & 4,568,367       \\
    Malayalam           & mal       & ml          & Southern Dravidian      & Dravidian      & Eurasia   & 3,662,571,498    \\
    Marathi             & mar       & mr          & Indic                   & Indo-European  & Eurasia   & 1,775,483,122    \\
    Nepali              & nep       & ne          & Indic                   & Indo-European  & Eurasia   & 2,551,307,393    \\
    Northern Sotho      & nso       & nso         & Bantoid                 & Niger-Congo    & Africa    & 1,764,506       \\
    Odia                & ori       & or          & Indic                   & Indo-European  & Eurasia   & 1,157,100,133    \\
    Portuguese          & por       & pt          & Romance                 & Indo-European  & Eurasia   & 79,277,543,375   \\
    Punjabi             & pan       & pa          & Indic                   & Indo-European  & Eurasia   & 1,572,109,752    \\
    Sesotho             & sot       & st          & Bantoid                 & Niger-Congo    & Africa    & 751,034        \\
    Setswana            & tsn       & tn          & Bantoid                 & Niger-Congo    & Africa    & 1,502,200       \\
    Simplified Chinese  &     ---      & zhs         & Chinese                 & Sino-Tibetan   & Eurasia   & 261,019,433,892  \\
    Spanish             & spa       & es          & Romance                 & Indo-European  & Eurasia   & 175,098,365,045  \\
    Swahili             & swh       & sw          & Bantoid                 & Niger-Congo    & Africa    & 236,482,543     \\
    Tamil               & tam       & ta          & Southern Dravidian      & Dravidian      & Eurasia   & 7,989,206,220    \\
    Telugu              & tel       & te          & South-Central Dravidian & Dravidian      & Eurasia   & 2993407,159    \\
    Traditional Chinese &      ---     & zht         & Chinese                 & Sino-Tibetan   & Eurasia   & 762,489,150     \\
    Twi                 & twi       & tw          & Kwa                     & Niger-Congo    & Africa    & 1,265,041       \\
    Urdu                & urd       & ur          & Indic                   & Indo-European  & Eurasia   & 2,781,329,959    \\
    Vietnamese          & vie       & vi          & Viet-Muong              & Austro-Asiatic & Eurasia   & 43,709,279,959   \\
    Wolof               & wol       & wo          & Wolof                   & Niger-Congo    & Africa    & 3,606,973       \\
    Xitsonga            & tso       & ts          & Bantoid                 & Niger-Congo    & Africa    & 707,634        \\
    Yoruba              & yor       & yo          & Defoid                  & Niger-Congo    & Africa    & 89,695,835      \\"""


    records = []
    for entry in mystring.split("\n"):
        entry = entry.replace("\\", "").strip()
        mylist = entry.split("&")
        language = mylist[0].strip()
        number = int(mylist[6].strip().replace(",",""))
        # print(f"{language} - {number}")
        records.append((language, number))
        
    dist_dict = {}
    records  = sorted(records, key=lambda x: x[1], reverse=True)
    top10 = records[:10]
    tot_top_10 = sum(e[1] for e in top10)
    for lang, freq in records[:10]:
        dist_dict[lang] = round(freq / tot_top_10, 2)
        
    return dist_dict

def tokenize_c4():
    parser = argparse.ArgumentParser(
        description="Stream & tokenize one C4 language up to a token budget"
    )
    parser.add_argument(
        "--lang", required=True,
        help="Language code (e.g., en, zh, fr, ...)"
    )
    parser.add_argument(
        "--total_tokens", type=int, default=700_000_000,
        help="Total tokens across all languages"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1024,
        help="Max tokens per example (truncation length)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="How many texts to batch-tokenize at once"
    )
    args = parser.parse_args()

    # per‐language ratios
    lang_ratios = {
        'en': 0.35, 'zh': 0.19, 'fr': 0.15, 'es': 0.13,
        'pt': 0.06, 'ar': 0.05, 'vi': 0.03, 'hi': 0.02,
        'id': 0.01, 'bn': 0.01
    }
    if args.lang not in lang_ratios:
        raise ValueError(f"Unknown lang code: {args.lang}")

    # compute this lang's target
    target = int(args.total_tokens * lang_ratios[args.lang])

    # let the Rust tokenizer use all cores
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(
        "bigscience/bloom-1b1", use_fast=True
    )

    path_to_c4 = "allenai___c4" # TODO: change this to where it is cached
    stream = load_dataset(
        path_to_c4,
        data_dir=args.lang,
        split="train",
        streaming=True
    )

    out = []

    pbar = tqdm(total=target, unit="tok", desc=args.lang)
    buffer = []
    cnt = 0

    def flush_batch():
        nonlocal cnt
        enc = tokenizer(buffer, add_special_tokens=False, truncation=False)
        for ids in enc["input_ids"]:
            if len(ids) < args.chunk_size:
                continue
            ids = ids[: args.chunk_size]
            out.append({"input_ids": ids})
            cnt += len(ids)
            pbar.update(len(ids))
            if cnt >= target:
                return True
        return False

    # streaming loop
    for ex in stream:
        txt = ex["text"]
        # cheap char-based prefilter to skip way-too-short docs
        if len(txt) < 2 * args.chunk_size:
            continue

        buffer.append(txt)
        if len(buffer) >= args.batch_size:
            done = flush_batch()
            buffer = []
            if done:
                break

    # flush any remaining buffered texts
    if cnt < target and buffer:
        flush_batch()

    pbar.close()
    
    print(f"[{args.lang}] done → {cnt} / {target} = pct: {cnt / target:.2%}")
    print(len(out), "sequences found.")
    
    # Create a Dataset object from the filtered data
    dataset = Dataset.from_list(out)
    data_dir = "workspace/c4cache"
    os.makedirs(data_dir, exist_ok=True)
    dataset.save_to_disk(f"{data_dir}/c4_bloom_{args.lang}.hf")
        
    # Step 2: Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    
    # Step 3: Split the dataset into train, validation, and test sets
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

    # Combine splits into a DatasetDict
    split_dataset = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    # Step 4: Save the splits locally
    hf_cache_dir = f"{data_dir}/c4_bloom_{args.lang}_split.hf"
    split_dataset.save_to_disk(hf_cache_dir)


def tokenize_and_filter_bloom_c4():
    # adjust this to wherever you’re keeping your per-lang splits
    # lang_data_dir = "workspace/c4cache"
    # save_data_dir = "workspace/cache"
    
    ############################################################################
    # STEP 1: Get language ratios
    lang_ratios = {
        'en': 0.35,
        'zh': 0.19,
        'fr': 0.15,
        'es': 0.13,
        'pt': 0.06,
        'ar': 0.05,
        'vi': 0.03,
        'hi': 0.02,
        'id': 0.01,
        'bn': 0.01
    }

    ############################################################################
    # STEP 2: Tokenize c4 and get data from each language in that ratio
    # Checkout tokenize_c4 function for this.

    ################################################################################
    # STEP 3: Split the dataset into train/test/val

    # from datasets import load_from_disk, concatenate_datasets, DatasetDict

    # # list of language codes you’ve processed
    # langs = ["ar", "bn", "en", "es", "fr", "hi", "id", "pt", "vi", "zh"]

    # train_dsets = []
    # val_dsets   = []
    # test_dsets   = []

    # for lang in langs:
    #     path = f"{lang_data_dir}/c4_bloom_{lang}_split.hf"
    #     ds = load_from_disk(path)

    #     train_dsets.append(ds["train"])
    #     val_dsets.append(ds["validation"])
    #     test_dsets.append(ds["test"])

    # # concatenate and shuffle
    # multilingual_train = concatenate_datasets(train_dsets).shuffle(seed=42)
    # multilingual_val   = concatenate_datasets(val_dsets).shuffle(seed=42)
    # multilingual_test  = concatenate_datasets(test_dsets).shuffle(seed=42)


    # # build a new DatasetDict
    # multilingual = DatasetDict({
    #     "train":      multilingual_train,
    #     "validation": multilingual_val,
    #     "test": multilingual_test
    # })

    # # save to disk
    # out_path = f"{save_data_dir}/c4_bloom_multilingual_split.hf"
    # multilingual.save_to_disk(out_path)

    # print(f"Multilingual split saved to {out_path}")


    ################################################################################
    # STEP 4: Uniform sample the train set top10 langs for annotation 
    #         where there is overlap with multiblimp
    from datasets import load_from_disk, concatenate_datasets, DatasetDict

    # list of language codes you’ve processed
    langs = ["en", "fr", "es", "pt", "ar", "hi"]

    train_dsets = []

    for lang in langs:
        path = f"{lang_data_dir}/c4_bloom_{lang}_split.hf"
        ds = load_from_disk(path)
        train_dsets.append(ds["train"].select(range(100)))

    # concatenate and shuffle
    multilingual_train = concatenate_datasets(train_dsets).shuffle(seed=42)

    # build a new DatasetDict
    multilingual = DatasetDict({
        "train":      multilingual_train,
    })

    # save to disk
    out_path = f"{save_data_dir}/c4_bloom_multilingual_split_uniform_multiblimp.hf"
    multilingual.save_to_disk(out_path)

    print(f"Uniform train split saved to {out_path}")


################################################################################
# Uploading & Downloading all from HF
################################################################################
def _folders_to_repos(hf_username, folders):
    """
    Map local '.hf' folder names -> Hub repo_ids (strip '.hf').
    Example: 'foo_bar.hf' -> 'username/foo_bar'
    """
    return {
        folder_name: f"{hf_username}/{folder_name[:-3]}" if folder_name.endswith(".hf")
        else f"{hf_username}/{folder_name.rstrip('/')}"
        for folder_name in folders
    }

def upload_all_to_hf(
    hf_username,
    local_base: str = os.path.join(".", "workspace", "cache"),
    folders = LMODELING_FOLDERS,
    private = False,
    commit_message = "Upload dataset snapshot",
    dry_run = False,
) -> None:
    """
    Upload all local '.hf' dataset folders to the Hugging Face Hub as dataset repos.
    Each local folder must be a directory produced by datasets.save_to_disk(...).

    Auth: run `huggingface-cli login` once locally or set env HUGGINGFACE_HUB_TOKEN.
    """
    api = HfApi()
    repos = _folders_to_repos(hf_username, folders)

    os.makedirs(local_base, exist_ok=True)

    for folder_name, repo_id in repos.items():
        print("-" * 20)
        local_path = os.path.join(local_base, folder_name)
        if os.path.isdir(local_path):
            print(f"Found directory: {local_path}")
        else:
            print(f"Skip: not found or not a directory: {local_path}")
            continue

        # quick sanity: looks like an Arrow dataset dir?
        has_arrow = any(
            os.path.exists(os.path.join(local_path, "train", p))
            for p in ("dataset_info.json", "state.json")
        )
        if not has_arrow:
            print(f"    Warning: {local_path} doesn't look like a datasets.save_to_disk() folder.")

        print(f"    Preparing repo: {repo_id}")
        if not dry_run:
            create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        if dry_run:
            continue
        
        print(f"    Uploading {local_path} => {repo_id}")
        try:
            # upload_folder will recurse and LFS large files automatically
            upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            print(f"Uploaded: {folder_name} → {repo_id}")
        except Exception as e:
            print(f"Failed to upload {folder_name}: {e}")

    print("All upload pass complete.")

def download_all_from_hf(
    hf_username,
    folders = LMODELING_FOLDERS,
    local_base = os.path.join(".", "workspace", "cache"),
    revision = None,  # e.g. "v1.0" or a commit hash, if you tag
) -> None:
    """
    Download all dataset repos into local '.hf' folders where your loader expects them.
    """
    HF_DATASETS = _folders_to_repos(hf_username, folders)

    os.makedirs(local_base, exist_ok=True)
    for folder_name, repo_id in HF_DATASETS.items():
        print("-" * 20)
        local_path = os.path.join(local_base, folder_name)
        if os.path.exists(local_path):
            print(f"Dataset already cached: {local_path}")
            continue

        print(f"Downloading {repo_id} => {local_path}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=local_path,
                local_dir_use_symlinks=False,  # real files so load_from_disk works as-is
            )
            print(f"Finished: {folder_name}")
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")

    print("All LModeling datasets have been downloaded.")

if __name__ == "__main__":
    # 1) Upload all LModeling folders in workspace/cache
    # upload_all_to_hf(hf_username="bayazitdeniz", dry_run=False)

    # 2) Or download everything from the Hub into the same locations
    download_all_from_hf(hf_username="bayazitdeniz")
