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
conda activate multiethnic

echo "The environment has been activated."

python -u multiomics/main.py --data_Category Race --omicsConfiguration combination --DDP_group BLACK --cancer_type PanGyn --endpoint OS --years 1 --features_count 100 --FeatureMethod 2 --omics_feature Protein_mRNA_MicroRNA_Methylation

echo "The execution has been done."


