#!/bin/bash
#SBATCH --job-name=cbramod_dl
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/dl_%j.out

cd /home/mat/projects/EEG_psychostimulant
source .venv/bin/activate

python eeg_adhd_epilepsy_psychostimulant/dl/run_dl_pipe.py \
  --config eeg_adhd_epilepsy_psychostimulant/dl/config_dl_filtered.yml
