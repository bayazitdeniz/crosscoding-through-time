#!/bin/bash

sae_dim=16384
model_name="bigscience/bloom-1b1-intermediate"
seed_list=$"124 153 6582"
layer=$"13"

################################################################################
# 2-way Comparison
################################################################################
revision_lists=$"global_step1000-global_step10000 \
global_step10000-global_step100000 global_step100000-main"

for seed in $seed_list;
    do
    for revision_list in $revision_lists;
        do
        job_name=$"train-bloom-$revision_list-$sae_dim-$seed-$layer"
        job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
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
revision_lists=$"global_step10000-global_step100000-main"

for seed in $seed_list;
    do
    for revision_list in $revision_lists;
        do
        job_name=$"train-bloom-3compar-$sae_dim-$seed"
        job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
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
