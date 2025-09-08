#!/bin/bash

# List of versions whose config will open in vscode
versions=(387 447 446)
mypath="workspace/logs/checkpoints/version_"

for version in $versions; do
    config_path=${mypath}${version}/20_cfg.json
    code $config_path
done