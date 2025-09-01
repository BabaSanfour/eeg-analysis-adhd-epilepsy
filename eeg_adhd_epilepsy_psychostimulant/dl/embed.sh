#!/bin/bash

#SBATCH --account=def-kjerbi
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca
#SBATCH --mail-type=ALL
#SBATCH --time=2:40:00
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --mem=20G
#SBATCH --output=embed.out
#SBATCH --error=embed.err
#SBATCH --job-name=embed

source /home/hamza97/venv_eeg_psychostim/bin/activate

python -m eeg_adhd_epilepsy_psychostimulant.dl.reve_model \
  --derivatives_dir "/home/hamza97/projects/rrg-shahabkb/cocolab_data/EEG_ADHD_epilepsy_psychostimulants/derivatives/embedding_processing" \
  --embeddings_dir "/home/hamza97/projects/rrg-shahabkb/cocolab_data/EEG_ADHD_epilepsy_psychostimulants/embeddings" \
  --n_subjects 1013 --start_subject 1 --segment_duration 10 --z_score --z_score_axis 1

python -m eeg_adhd_epilepsy_psychostimulant.dl.reve_model \
  --derivatives_dir "/home/hamza97/projects/rrg-shahabkb/cocolab_data/EEG_ADHD_epilepsy_psychostimulants/derivatives/embedding_processing" \
  --embeddings_dir "/home/hamza97/projects/rrg-shahabkb/cocolab_data/EEG_ADHD_epilepsy_psychostimulants/embeddings" \
  --n_subjects 1013 --start_subject 1 --segment_duration 10 --z_score --z_score_axis 0

