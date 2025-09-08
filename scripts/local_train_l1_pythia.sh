#!/bin/bash

sae_dim=16384
model_name="EleutherAI/pythia-1b"
seed_list=$"6582"
layer_list=$"9"

################################################################################
# 2-way Comparison
################################################################################
revision_lists=$"step64-step512 step512-step2000 step2000-main"

for seed in $seed_list;
    do
    for revision_list in $revision_lists;
        do
        for layer in $layer_list;
            do
            job_name=$"pythia-$revision_list-$seed"
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
    done

################################################################################
# 3-way Comparison
################################################################################
revision_lists=$"step512-step2000-main"

for seed in $seed_list;
    do
    for revision_list in $revision_lists;
        do
        for layer in $layer_list;
            do
            job_name=$"pythia-3compar-$seed"
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
    done
