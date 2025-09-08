#!/bin/bash

port=8089
ckpt_num=$"20"
project_path="./workspace/logs/checkpoints"
threshold_list=$"0.1"
max_examples=1000
batch_size=4
top_k_list="100 10"
vis_with_lmodeling=True

datasets="distractor_agreement_relational_noun \
distractor_agreement_relative_clause \
irregular_plural_subject_verb_agreement_1 \
regular_plural_subject_verb_agreement_1"

version_list=(218 265 219)

for top_k in $top_k_list; do
    for dataset_name in $datasets; do
        for version_num in $version_list; do
            for node_threshold in $threshold_list; do

                task=subjectverb-$dataset_name
                job_name=$"pythia-$version_num-$task-topk$top_k"
                job_name=${job_name//irregular_plural_subject_verb_agreement_1/irregular}
                job_name=${job_name//regular_plural_subject_verb_agreement_1/regular}
                job_name=${job_name//distractor_agreement_relational_noun/relationalnoun}
                job_name=${job_name//distractor_agreement_relative_clause/relativeclause}
                job_name=$(sed "s/[^[:alnum:]-]//g" <<< "$job_name")
                echo $job_name
                
                python \
                    "attribution.py" \
                    --project_path=$project_path \
                    --version_num="version_$version_num" \
                    --ckpt_num=$ckpt_num \
                    --dataset_name=$task \
                    --node_threshold=$node_threshold \
                    --do_threshold=True \
                    --max_examples=$max_examples \
                    --batch_size=$batch_size \
                    --just_ablation=True --indiv_feat_ablation=False \
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
done
