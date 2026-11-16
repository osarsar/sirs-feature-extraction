#!/bin/bash
#SBATCH --job-name=arch_ablation
#SBATCH --account=cyber_cc-lcbfvhtc9qm-premium-gpu
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/oussama.sarsar/paper2/ablation_%j.log
#SBATCH --error=/home/oussama.sarsar/paper2/ablation_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

PYTHON=/home/oussama.sarsar/.conda/envs/wahed/bin/python
CONDA_ENV=/home/oussama.sarsar/.conda/envs/wahed
LOCAL_PKGS=/tmp/$SLURM_JOB_ID/site-packages

# Copy critical packages to local /tmp to avoid Remote I/O errors
echo "Copying packages to local storage..."
mkdir -p $LOCAL_PKGS
cp -r $CONDA_ENV/lib/python3.10/site-packages/scipy $LOCAL_PKGS/ 2>/dev/null
cp -r $CONDA_ENV/lib/python3.10/site-packages/scipy.libs $LOCAL_PKGS/ 2>/dev/null
cp -r $CONDA_ENV/lib/python3.10/site-packages/sklearn $LOCAL_PKGS/ 2>/dev/null
cp -r $CONDA_ENV/lib/python3.10/site-packages/scikit_learn* $LOCAL_PKGS/ 2>/dev/null
echo "Copy done."

# Prepend local packages to PYTHONPATH so they load from /tmp
export PYTHONPATH=$LOCAL_PKGS:$PYTHONPATH

cd ~/paper2

# Verify
echo "Training data: $($PYTHON -c "import json; d=json.load(open('data/final/training_data_with_dimensions.json')); print(len(d), 'claims')")"
echo "Gold test: $($PYTHON -c "import json; d=json.load(open('data/final/gold_test_with_dimensions.json')); print(len(d), 'claims')")"

# Test scipy import
$PYTHON -c "from scipy.stats import spearmanr; print('scipy import OK')"

mkdir -p results

echo "Starting architecture ablation..."
$PYTHON scripts/architecture_ablation.py

echo "Job finished: $(date)"
