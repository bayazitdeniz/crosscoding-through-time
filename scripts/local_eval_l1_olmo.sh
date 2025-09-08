#!/bin/bash

# version_list=(430 431 442 434 435 428 436 438 441)
# version_list=(440 439 429 433 432 437)
version_list=(447 460 459)

project_path="./workspace/logs/checkpoints"

for i in $version_list;
    do
    project_name=$"version_$i"
    
    job_name=$"eval-olmo-$project_name"
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

