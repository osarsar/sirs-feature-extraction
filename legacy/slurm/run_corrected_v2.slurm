#!/bin/bash
#SBATCH --job-name=corrected_train
#SBATCH --account=cyber_cc-lcbfvhtc9qm-premium-gpu
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/oussama.sarsar/paper2/corrected_%j.log
#SBATCH --error=/home/oussama.sarsar/paper2/corrected_%j.err

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

export PYTHONPATH=$LOCAL_PKGS:$PYTHONPATH

cd ~/paper2

$PYTHON -c "from scipy.stats import spearmanr; print('scipy OK')"

mkdir -p results

echo "Starting corrected training (separate + unified, 3 seeds)..."
$PYTHON scripts/train_corrected.py

echo "Job finished: $(date)"
