#!/bin/bash
#SBATCH --job-name=multiomics
#SBATCH --output=multiomics/logs/task_%a.out
#SBATCH --error=multiomics/logs/task_%a.err
#SBATCH --array=0-3199%50
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --gres=gpu:1
#SBATCH -A isaac-uthsc0057

module load anaconda3/2024.06
source $ANACONDA3_SH
conda activate multiethnic

echo "Task $SLURM_ARRAY_TASK_ID started at $(date)"

python -u multiomics/main.py \
    --task_file multiomics/task_list_multiomics/tasks.json \
    --task_index $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"
