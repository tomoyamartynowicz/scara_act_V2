#!/bin/bash

#SBATCH --job-name=act_leaf_rgb
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=education-ME-msc-ro

set -euo pipefail

module purge
module load slurm

export PATH="$HOME/.pixi/bin:$PATH"

PIXI_PROJECT="/scratch/$USER/thesis/envs/act"
DATASET_DIR="/scratch/$USER/thesis/datasets/leaf_cutting_experiment_rgb"
CKPT_DIR="/scratch/$USER/thesis/runs/act/leaf_rgb_act_v1"

mkdir -p "$CKPT_DIR"

cd "$HOME/scara_ws/src/scara_act"

echo "Node: $(hostname)"
echo "Dataset: $DATASET_DIR"
echo "Checkpoint dir: $CKPT_DIR"

srun pixi run \
  --manifest-path "$PIXI_PROJECT/pixi.toml" \
  python -u imitate_episodes_resume.py \
  --dataset_dir "$DATASET_DIR" \
  --ckpt_dir "$CKPT_DIR" \
  --num_epochs 100 \
  --batch_size 8