#!/bin/bash
cd /mnt/hdd16t/workspace/anchor
source $(conda info --base)/etc/profile.d/conda.sh
conda activate anchor
python -u scripts/attribution/eval_comparison_table.py --max-samples 100 --steps 10
