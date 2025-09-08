#!/bin/bash

model_name=$"allenai/OLMo-1B-0724-hf"
task=$"blimp"
all_steps=$@

echo "The following steps will be evaluated:"
echo $all_steps

for revision in $all_steps; do
    lm_eval \
        --model hf \
        --model_args pretrained=$model_name,dtype=float16,revision=${revision} \
        --tasks $task \
        --device cuda:0 \
        --batch_size 8 \
        --num_fewshot 0 \
        --output_path "workspace/results/lm_eval_harness_res" \
        --log_samples
done