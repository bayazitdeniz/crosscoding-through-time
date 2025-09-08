import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_revision_nnsight

def prepare_bloom_revision_list():
    steps = [1_000, 10_000, 100_000, 200_000, 300_000, 500_000, 600_000]
    revision_list = ["global_step" + str(step) for step in steps]
    print(revision_list)
    print([revision2tokens_bloom(rev) for rev in revision_list])
    print(len(revision_list))
    return revision_list

def prepare_eval_dataset(dataset_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_task_dataset(dataset_name, tokenizer=tokenizer)["train"]
    num_examples = len(dataset)
    n_batches = math.floor(num_examples / batch_size)
    batches = [
        dataset[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    return batches

@torch.no_grad()
def multiblimp_harness(model_name, revs, batches):
    results = {}
    device="cuda:0"
    for rev in tqdm(revs):
        model = load_revision_nnsight(
            model_name=model_name,
            revision_list=[rev],
            device=device,
            seed=42
        )[0]
        
        res = 0.0
        tot = 0.0
        for idx, batch in enumerate(batches):
            model_input = batch["clean_prefix"]
            prob_dict = {}
            with model.trace(model_input):
                model.output.logits[:, -1, :]                
                for answer_type in ["clean_answer", "patch_answer"]:
                    answer_toks = torch.tensor(
                        model.tokenizer(batch[answer_type]).input_ids,
                        dtype=torch.long,
                        device=device
                    ).squeeze(-1)
                    
                    prob = torch.gather(
                        model.output.logits[:, -1, :],
                        dim=-1,
                        index=answer_toks.view(-1, 1),
                    ).squeeze(-1)
                    
                    prob_dict[answer_type] = prob.save()
            
            res += (prob_dict["clean_answer"] > prob_dict["patch_answer"]).sum().item()
            tot += len(prob_dict["clean_answer"])  
        print(res)
        print(tot)      
        results[rev] = res / tot
            
    filename = f"workspace/act_cache/accuracy_{model_name.replace('/','_')}.json"
    new_results = {str(k): v for k, v in results.items()}
    json_dict = json.dumps(new_results,  ensure_ascii=False, indent=4) 
    f = open(filename, "w")
    f.write(json_dict)
    f.close()
            
    return results

def main()
    layer_num = 12
    model_name = "bigscience/bloom-1b1-intermediate"
    batch_size = 32
    revision_list = prepare_bloom_revision_list()

    full_dict = {}
    for lang in ["eng", "fra", "spa", "por", "arb", "hin"]:
        for task in ["SV-#", "SV-G", "SV-P"]:
            acc = None
            dataset_name = f"multiblimp_{lang}_{task}"
            batches = prepare_eval_dataset(dataset_name, model_name="bigscience/bloom-1b1")
            if len(batches) != 0:
                acc = multiblimp_harness(model_name, revision_list, batches)
            if lang not in full_dict:
                full_dict[lang] = {}
            if task not in full_dict[lang]:
                full_dict[lang][task] = {}
            full_dict[lang][task] = acc
            print(acc)

    filename = f"workspace/results/lm_eval_harness_res/bloom1b_multiblimp_performance.json"
    json_dict = json.dumps(full_dict,  ensure_ascii=False, indent=4) 
    f = open(filename, "w")
    f.write(json_dict)
    f.close()

if __name__ == "__main__":
    main()