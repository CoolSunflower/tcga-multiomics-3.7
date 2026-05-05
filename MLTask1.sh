#!/bin/bash
#SBATCH --job-name=MLTask1.txt
#SBATCH --output=multiomics/o_e_files/MLTask1.o%j
#SBATCH --error=multiomics/o_e_files/MLTask1.e%j
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A isaac-uthsc0057
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --gres=gpu:1

# Load any necessary modules
module load anaconda3/2024.06
source $ANACONDA3_SH
conda activate multiethnic_3_7

echo "The environment has been activated."

python -u tcga-multiomics-3.7/main.py --task_file tcga-multiomics-3.7/task_list_multiomics/tasks.json --task_index 0

echo "The execution has been done."


