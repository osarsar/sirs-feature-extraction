#!/bin/bash
# Toubkal (UM6P) HPC job script. Adapt account, partition and paths to your cluster.
#SBATCH --job-name=p2_v4_pred
#SBATCH --account=cyber_cc-lcbfvhtc9qm-premium-gpu
#SBATCH --partition=gpu_h100
#SBATCH --qos=premium-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/srv/lustre01/project/cyber_cc-lcbfvhtc9qm/users/oussama.sarsar/logs_gpu/p2_v4_%j.log
#SBATCH --error=/srv/lustre01/project/cyber_cc-lcbfvhtc9qm/users/oussama.sarsar/logs_gpu/p2_v4_%j.err

# Paper 2 v4: v3 + save val/test predictions for weight optimisation.
echo "Job started: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

PYTHON=/home/oussama.sarsar/.conda/envs/wahed/bin/python
CONDA_ENV=/home/oussama.sarsar/.conda/envs/wahed
LOCAL_PKGS=/tmp/$SLURM_JOB_ID/site-packages
mkdir -p $LOCAL_PKGS
cp -r $CONDA_ENV/lib/python3.10/site-packages/scipy $LOCAL_PKGS/ 2>/dev/null
cp -r $CONDA_ENV/lib/python3.10/site-packages/scipy.libs $LOCAL_PKGS/ 2>/dev/null
cp -r $CONDA_ENV/lib/python3.10/site-packages/sklearn $LOCAL_PKGS/ 2>/dev/null
cp -r $CONDA_ENV/lib/python3.10/site-packages/scikit_learn* $LOCAL_PKGS/ 2>/dev/null
export PYTHONPATH=$LOCAL_PKGS:$PYTHONPATH

cd ~/paper2
$PYTHON -c "from scipy.stats import spearmanr; import sys; sys.path.insert(0,'scripts'); from sirs_paper1 import FeatureExtractor; print('imports OK')"
$PYTHON scripts/train_corrected_v4.py
echo "Job finished: $(date)"
