#!/bin/bash
set -euo pipefail

# List of versions to remove
versions=(137 47 15)

for n in "${versions[@]}"; do
  dir="version_$n"
  if [ -d "$dir" ]; then
    rm -rf "$dir"
    echo "Deleted $dir"
  else
    echo "Skipping $dir (not found)"
  fi
done