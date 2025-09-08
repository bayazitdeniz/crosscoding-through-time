#!/bin/bash

task="blimp"
model_name="pythia1b"

job_name_subset=$"lmeval-$task-$model_name"
job_name_subset=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name_subset")
job_name_subset=$(echo "$job_name_subset" | tr '[:upper:]' '[:lower:]')

################################################################################
job_name="$job_name_subset-1"
echo $job_name

revision_list=$"0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 7000 8000 9000"
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list

################################################################################
job_name="$job_name_subset-2"
echo $job_name

revision_list=$(seq 10000 1000 29000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list


################################################################################
job_name="$job_name_subset-3"
echo $job_name

revision_list=$(seq 30000 1000 49000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list

################################################################################
job_name="$job_name_subset-4"
echo $job_name

revision_list=$(seq 50000 1000 69000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list

################################################################################
job_name="$job_name_subset-5"
echo $job_name

revision_list=$(seq 70000 1000 89000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list


################################################################################
job_name="$job_name_subset-6"
echo $job_name

revision_list=$(seq 90000 1000 109000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list


################################################################################
job_name="$job_name_subset-7"
echo $job_name

revision_list=$(seq 110000 1000 129000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list

################################################################################
job_name="$job_name_subset-8"
echo $job_name

revision_list=$(seq 130000 1000 143000)
bash scripts/rcp_lm_harness_pythia_inner.sh $revision_list