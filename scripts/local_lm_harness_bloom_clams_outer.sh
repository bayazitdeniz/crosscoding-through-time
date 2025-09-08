#!/bin/bash

task=$"clams"
model_name="bloom1b"

job_name_subset=$"lmeval-$task-$model_name"
job_name_subset=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name_subset")
job_name_subset=$(echo "$job_name_subset" | tr '[:upper:]' '[:lower:]')
run_num=1

all_steps=$"global_step1000 \
global_step10000 global_step100000 \
global_step200000 global_step300000 \
global_step400000 global_step500000 global_step600000 main"

for revision in $all_steps; do
    job_name="$job_name_subset-$run_num"
    echo $job_name
    
    bash scripts/rcp_lm_harness_bloom_inner.sh $revision
    ((run_num++))
done