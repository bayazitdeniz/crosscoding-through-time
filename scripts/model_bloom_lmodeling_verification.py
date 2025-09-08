from itertools import chain
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import datasets

def group_texts(examples, block_size):
    # Concatenate all inputs
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop the small remainder
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Copy the same for labels
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_and_chunk(
        wikitext_dataset,
        column_names,
        tokenizer,
        block_size=None
    ):
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # 1) Decide on block size depending on model context length if None is given
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024
    else:
        block_size = min(block_size, tokenizer.model_max_length)
    
    # 2) Tokenize texts
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        desc="Running tokenizer on ControlLM dataset"
    )

    # 3) Chunk token IDs into blocks
    controllm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(
            examples=examples, 
            block_size=block_size
        ),
        batched=True,
        num_proc=1,
        desc=f"Grouping texts in chunks of {block_size}"
    )
    
    return controllm_datasets

def load_wikitext2_test_dataloader(
    lm_name: str,
    block_size=512,
    controllm_eval_batch_size: int = 8
):
    # 1) Load the dataset
    wikitext_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10%]")
    # print(wikitext_dataset.info.version)
    
    # 2) Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    column_names = list(wikitext_dataset.features)
    
    # 3) Tokenize and chunk
    lm_datasets = tokenize_and_chunk(
        wikitext_dataset,
        column_names,
        tokenizer,
        block_size
    )
    lm_datasets.tokenizer = tokenizer
    
    print("Test ControlLM len: ", len(lm_datasets))

    # 4) Create test dataloader
    test_dataloader = DataLoader(
        lm_datasets, 
        shuffle=False,
        collate_fn=default_data_collator, 
        batch_size=controllm_eval_batch_size
    )

    return test_dataloader

@torch.no_grad()
def get_wiki_ppl(
    model, 
    wiki_dataloader: DataLoader, 
):
    sum_ppl = 0.0
    cnt = 0
    for batch in wiki_dataloader:
        batch["input_ids"] = batch["input_ids"].cuda().detach()
        batch["attention_mask"] = batch["attention_mask"].cuda().detach()
        tok_labels = batch["labels"].cuda().detach()

        output = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=tok_labels)
        loss = output.loss.detach()
        
        sum_ppl += torch.exp(loss).item()
        cnt += 1

    return sum_ppl / float(cnt)


dataloader = load_wikitext2_test_dataloader(lm_name="bigscience/bloom-1b1")
steps = [1_000, 10_000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000]
resdict = {}
for step in tqdm(steps, desc="steps"):
    rev = f"global_step{step}"
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-1b1-intermediate",
        revision=rev
    )
    model = model.cuda()
    ppl = get_wiki_ppl(model, dataloader)
    resdict[step] = ppl

print(resdict)

# Output resdict:
# {1000: 299.0409986707899,
#  10000: 53.00332175360786,
#  100000: 34.277858310275604,
#  200000: 32.3691151936849,
#  300000: 30.913375430636936,
#  400000: 338341.52430555556, <------ very high
#  500000: 28.414903428819443,
#  600000: 26.762985229492188}