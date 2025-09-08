#!/bin/bash

model_name=$"bigscience/bloom-1b1-intermediate"
task=$"clams"
tokenizer=$"bigscience/bloom-1b1"
all_steps=$@

echo "The following steps will be evaluated:"
echo $all_steps

for revision in $all_steps; do
    lm_eval \
        --model hf \
        --model_args pretrained=${model_name},dtype=float16,revision=${revision},tokenizer=${tokenizer} \
        --tasks $task \
        --include_path "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bayazit/model-diff/clams_data" \
        --device cuda:0 \
        --batch_size 8 \
        --num_fewshot 0 \
        --output_path "workspace/results/lm_eval_harness_res"
done