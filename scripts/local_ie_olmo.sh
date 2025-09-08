#!/bin/bash

port=8080
ckpt_num=$"20"
project_path="./workspace/logs/checkpoints"
node_threshold=$"0.1"
max_examples=1000
batch_size=4
top_k=10
vis_with_lmodeling=True

# version_list=(430 434 436)
# version_list=(440 433)
version_list=(447)
# version_list=(1064)

datasets=$"subjectverb"

for dataset_name in $datasets; do
    for version_num in $version_list; do
        job_name=$"ieig-olmo-$version_num-$dataset_name-$top_k"
        job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
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
            --vis_with_lmodeling=$vis_with_lmodeling #\
            # --bottom_and_top=True
        
        ((port++))
    done
done
