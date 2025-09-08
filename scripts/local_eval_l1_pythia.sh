#!/bin/bash

# version_list=(218 311 265 318 219 315)
# version_list=(443 444 445)
version_list=(446 455 456)

project_path="./workspace/logs/checkpoints"

for i in $version_list;
    do
    project_name=$"version_$i"
    
    job_name=$"eval-pythia-$project_name"
    job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
    echo $job_name

    python \
        "test_crosscoder_revision.py" \
        --project_name=$project_name \
        --project_path=$project_path \
        --do_ce_eval \
        --do_sparsity_eval \
        --last_ckpt_only
    done

