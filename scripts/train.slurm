#!/bin/bash
# OPTIONAL: train the YOLO detector on a SLURM GPU cluster.
# Only use this on a cluster you are allowed to use. On a laptop, `cobotsort train` (GPU) or
# `cobotsort train --quick` (CPU) does the same job.
#
# Usage: sbatch scripts/train.slurm   (from the repository root, after `cobotsort dataset`)
#SBATCH --job-name=cobotsort-yolo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out

set -euo pipefail
mkdir -p logs
# Activate the environment (pick one).
source .venv/bin/activate            # venv
# conda activate cobotsort           # or conda
export MUJOCO_GL=egl                 # headless rendering, if the dataset is generated on the node

cobotsort train --epochs 40 --imgsz 640 --device 0
