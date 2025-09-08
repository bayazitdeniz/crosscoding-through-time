#!/bin/bash

sae_dim=16384
model_name="allenai/OLMo-1B-0724-hf"
seed_list=$"124 153 6582"
layer=9

################################################################################
# 2-way Comparison
################################################################################
revision_lists=$"step1000-tokens2B_step2000-tokens4B \
    step2000-tokens4B_step16000-tokens33B \
    step16000-tokens33B_step137000-tokens287B \
    step16000-tokens33B_step1454000-tokens3048B \
    step137000-tokens287B_step1454000-tokens3048B"

for seed in $seed_list;
    do
    for revision_list in $revision_lists;
        do
        job_name=$"olmo-$revision_list-$seed"
        job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
        job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
        echo $job_name
        
        python \
            "train_crosscoder_revision.py" \
            --model_name=$model_name \
            --revision_list=$revision_list \
            --dict_size=$sae_dim \
            --seed=$seed \
            --wandb_name=$job_name \
            --hook_point=blocks.$layer.hook_resid_pre
        done
    done

################################################################################
# 3-way Comparison
################################################################################
revision_lists=$"step2000-tokens4B_step16000-tokens33B_step1454000-tokens3048B"

for seed in $seed_list;
    do
    for revision_list in $revision_lists;
        do
        job_name=$"olmoå-3compar-$seed"
        job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
        job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
        echo $job_name
        
        python \
            "train_crosscoder_revision.py" \
            --model_name=$model_name \
            --revision_list=$revision_list \
            --dict_size=$sae_dim \
            --seed=$seed \
            --wandb_name=$job_name \
            --hook_point=blocks.$layer.hook_resid_pre
        done
    done