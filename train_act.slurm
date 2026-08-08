#!/bin/bash

#SBATCH --job-name=act_leaf_rgb
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/scratch/thesis/runs/act/leaf_rgb_act_v1/slurm-%j.out
#SBATCH --account=education+

module load 2024r1
module load openmpi
module load py-torch

cd "$HOME/scara_ws/src/scara_act"

srun python -u imitate_episodes_resume.py \
  --dataset_dir /scratch/thesis/datasets/leaf_cutting_experiment_rgb \
  --ckpt_dir /scratch/thesis/runs/act/leaf_rgb_act_v1 \
  --num_epochs 2000 \
  --batch_size 8