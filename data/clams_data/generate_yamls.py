#!/usr/bin/env python3
"""
Generate one YAML file per JSONL + a suite file for lm‑eval.

Usage
-----
python generate_clams_yaml.py /path/to/jsonl_dir  \
       --suite_name clams --out_dir /path/to/yaml_out
"""

import argparse
from pathlib import Path
import yaml
import textwrap

TEMPLATE = {
    "dataset_path": "json",
    "output_type": "multiple_choice",
    "validation_split": "validation",
    "doc_to_text": '""',
    "doc_to_target": 0,
    "doc_to_choice": "{{[sentence_good, sentence_bad]}}",
    "metric_list": [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
    ],
    "num_fewshot": 0,
}

def one_yaml(file_path: Path, task_name) -> dict:
    cfg = {}
    cfg["task"] = task_name
    cfg["include"] = "_template.yaml"
    cfg["dataset_kwargs"] = {
        "data_files": {"validation": str(file_path.resolve())}
    }
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_dir", type=Path,
                    help="Folder containing *.jsonl task files")
    ap.add_argument("--suite_name", default="clams",
                    help="Name you will pass to --tasks")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to write YAMLs (defaults to jsonl_dir/yaml)")
    args = ap.parse_args()

    in_dir = args.jsonl_dir.expanduser().resolve()
    out_dir = (args.out_dir or in_dir / "yaml").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = []

    for jp in sorted(in_dir.glob("*.jsonl")):
        task_name = jp.stem                # agr_number, etc.
        task_names.append(task_name)

        yaml_path = out_dir / f"{task_name}.yaml"
        with yaml_path.open("w") as fh:
            yaml.safe_dump(one_yaml(jp, task_name), fh, sort_keys=False)

    # ---------- suite / group file ----------
    group_file = out_dir / f"{args.suite_name}_group.yaml"
    group_cfg = {"group": args.suite_name, "task": task_names}
    with group_file.open("w") as fh:
        yaml.safe_dump(group_cfg, fh, sort_keys=False)
        
    template_file = out_dir / "_template.yaml"
    with template_file.open("w") as fh:
        yaml.safe_dump(TEMPLATE, fh, sort_keys=False)

    print(f"✅  Wrote {len(task_names)} YAML task files and "
          f"group file → {out_dir}")

if __name__ == "__main__":
    main()
