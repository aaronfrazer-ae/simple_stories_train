#!/bin/bash
# Train the 30M model on one GPU of the AE Studio cluster.
# Submit from the repo root (~/simple_stories_train). Extra arguments go to train_llama.py:
#   sbatch slurm/train_30m.sbatch --num_iterations=300          (speed test)
#   sbatch --time=08:00:00 slurm/train_30m.sbatch               (full run, repo default 60,000 steps)
# --wckey, --time and --mem-per-gpu follow AE's GPU Cluster Best Practices.
#SBATCH --job-name=ss-30m
#SBATCH --wckey=alignment-onboarding
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/simple_stories_train"
nvidia-smi -L
../.venv/bin/python train_llama.py 30M_config.yaml --output_dir="$SLURM_SUBMIT_DIR/out" "$@"
echo "train_llama.py exited at $(date -u +%FT%TZ)"
