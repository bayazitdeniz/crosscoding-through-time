#!/bin/bash

port=8084
ckpt_num=$"20"
project_path="./workspace/logs/checkpoints"
node_threshold=$"0.1"
max_examples=5000
batch_size=4
top_k=10
vis_with_lmodeling=True

langs=$"eng fra spa por arb hin" # all
tasks="all"
# version_list=(387 400 409)
version_list=(454)
# version_list=(1059)

for lang in $langs; do
    for task in $tasks; do
        dataset_name="multiblimp_${lang}_${task}"
        for version_num in $version_list; do
            
            job_name=$"ieig-bloom-$version_num-$dataset_name"
            job_name="${job_name//_/-}"
            job_name="${job_name//#/n}"
            job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
            job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
            echo $job_name

            python \
                "attribution.py" \
                --project_path=$project_path \
                --version_num="version_$version_num" \
                --ckpt_num=$ckpt_num \
                --dataset_name=$dataset_name \
                --node_threshold=$node_threshold \
                --do_threshold=True \
                --max_examples=$max_examples \
                --batch_size=$batch_size \
                --ie_precomputed=False \
                --html_saved=False \
                --serve_html=False \
                --top_k=$top_k \
                --port=$port \
                --vis_with_lmodeling=$vis_with_lmodeling
            ((port++))
        done
    done
done