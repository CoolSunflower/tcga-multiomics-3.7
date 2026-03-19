# -*- coding: utf-8 -*-
"""
Task Script Generator for Multiomics Cancer ML Pipeline

This script generates SLURM job scripts and sbatch commands for running
multiomics cancer classification tasks.

TASK VARIABLES:
Fixed Parameters for a task:
- Race (fixed) (--data_Category)
- combination (fixed) (--omicsConfiguration)

Variables to loop over (via --run_all_feature_methods flag):
- features_count (100, 200)
- FeatureMethod (0, 1, 2: AutoencoderSettings 1 & 2) (total 4 methods x 2 counts = 8 configs)

Variables from file:
- DDP_group (BLACK, ASIAN, NAT_A) (from file or filename)
- cancer_type (from file)
- omics_feature (from file, 2-4 omic groups)
- endpoint (from file)
- years (from file)

CSV File Types:
1. Two-omics file: MLTasks_TwoFeaturesCombinations_Year02(Sheet1).csv
   - Contains Target Group column for DDP_group

2. Three/Four-omics files: {ddp_group}_{num}_omics.csv
   - DDP group extracted from filename (BLACK, ASIAN, NAT_A)
"""

import os
import pandas as pd
import re
from pathlib import Path


# Configuration
ROOT_FOLDER = "multiomics"
ENVIRONMENT_NAME = "myenv"
OUTPUT_DIR = "generated_scripts"
SBATCH_FILE = "sbatch_commands.sh"

# Script template
SCRIPT_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={RootFolder}/o_e_files/{job}.o%j
#SBATCH --error={RootFolder}/o_e_files/{job}.e%j
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --ntasks-per-node=8
#SBATCH -A isaac-uthsc0057
#SBATCH --cpus-per-task=8                  # CPUs per task
#SBATCH --mem=32G                           # Total memory
#SBATCH --partition=ai-tenn                   # Partition name
#SBATCH --qos=ai-tenn               # AI-TENN QoS
#SBATCH --gres=gpu:1

# Load any necessary modules
module load anaconda3/2024.06
source $ANACONDA3_SH
conda activate {ENVIRONMENT_NAME}

echo "The environment has been activated."

python -u {RootFolder}/main.py --data_Category Race --omicsConfiguration combination --DDP_group {DDP_GROUP} --cancer_type {CANCER_TYPE} --endpoint {ENDPOINT} --years {TIME} --omics_feature {FEATURE_STRING} --run_all_feature_methods

echo "The execution has been done."
'''


def parse_feature_type(feature_str):
    """
    Parse feature type string like "('Protein', 'mRNA', 'MicroRNA')" to "Protein_mRNA_MicroRNA"
    """
    # Remove parentheses and quotes
    cleaned = feature_str.strip("()' ")
    # Extract feature names
    features = re.findall(r"'([^']+)'", feature_str)
    if not features:
        # Try without quotes
        features = [f.strip() for f in cleaned.split(',')]
    return '_'.join(features)


def extract_ddp_group_from_filename(filename):
    """
    Extract DDP group from filename like 'black_three_omics.csv' -> 'BLACK'
    """
    basename = os.path.basename(filename).lower()
    if 'black' in basename:
        return 'BLACK'
    elif 'asian' in basename:
        return 'ASIAN'
    elif 'nat_a' in basename or 'nata' in basename:
        return 'NAT_A'
    return None


def generate_task_name(cancer_type, omics_feature, endpoint, years, ddp_group):
    """
    Generate a unique task name for the job.
    """
    feature_str = omics_feature.replace('_', '-')
    return f"MO_{ddp_group}_{cancer_type}_{feature_str}_{endpoint}_{years}YR"


def process_two_omics_file(filepath, output_dir):
    """
    Process the two-omics CSV file (MLTasks_TwoFeaturesCombinations_Year02).

    Columns: Cancer Type, Feature Type, Clinical Outcome Endpoint, Event Time Threshold (Year),
             Source Group, Target Group, EA-Positive, EA-Negative, DDP-Positive, DDP-Negative
    """
    print(f"Processing two-omics file: {filepath}")

    df = pd.read_csv(filepath)
    scripts = []
    sbatch_commands = []

    for idx, row in df.iterrows():
        cancer_type = row['Cancer Type']
        feature_type = row['Feature Type']
        endpoint = row['Clinical Outcome Endpoint']
        years = int(row['Event Time Threshold (Year)'])
        ddp_group = row['Target Group'].upper()

        # Parse feature type
        omics_feature = parse_feature_type(feature_type)

        # Generate task name
        task_name = generate_task_name(cancer_type, omics_feature, endpoint, years, ddp_group)

        # Generate script content
        script_content = SCRIPT_TEMPLATE.format(
            job=task_name,
            RootFolder=ROOT_FOLDER,
            ENVIRONMENT_NAME=ENVIRONMENT_NAME,
            DDP_GROUP=ddp_group,
            CANCER_TYPE=cancer_type,
            ENDPOINT=endpoint,
            TIME=years,
            FEATURE_STRING=omics_feature
        )

        script_filename = f"{task_name}.sh"
        script_path = os.path.join(output_dir, script_filename)

        scripts.append((script_path, script_content))
        sbatch_commands.append(f"sbatch {ROOT_FOLDER}/task_list_multiomics/{OUTPUT_DIR}/{script_filename}")

    return scripts, sbatch_commands


def process_three_four_omics_file(filepath, output_dir):
    """
    Process three/four-omics CSV files.

    Columns: Cancer_type, Feature_type, Target (endpoint), Years, 1WHITE, 0WHITE, 1{DDP}, 0{DDP}
    DDP group is extracted from the filename.
    """
    print(f"Processing three/four-omics file: {filepath}")

    ddp_group = extract_ddp_group_from_filename(filepath)
    if not ddp_group:
        print(f"Warning: Could not extract DDP group from filename: {filepath}")
        return [], []

    df = pd.read_csv(filepath)
    scripts = []
    sbatch_commands = []

    for idx, row in df.iterrows():
        cancer_type = row['Cancer_type']
        feature_type = row['Feature_type']
        endpoint = row['Target']
        years = int(row['Years'])

        # Parse feature type
        omics_feature = parse_feature_type(feature_type)

        # Generate task name
        task_name = generate_task_name(cancer_type, omics_feature, endpoint, years, ddp_group)

        # Generate script content
        script_content = SCRIPT_TEMPLATE.format(
            job=task_name,
            RootFolder=ROOT_FOLDER,
            ENVIRONMENT_NAME=ENVIRONMENT_NAME,
            DDP_GROUP=ddp_group,
            CANCER_TYPE=cancer_type,
            ENDPOINT=endpoint,
            TIME=years,
            FEATURE_STRING=omics_feature
        )

        script_filename = f"{task_name}.sh"
        script_path = os.path.join(output_dir, script_filename)

        scripts.append((script_path, script_content))
        sbatch_commands.append(f"sbatch {ROOT_FOLDER}/task_list_multiomics/{OUTPUT_DIR}/{script_filename}")

    return scripts, sbatch_commands


def main():
    """
    Main function to generate all task scripts and sbatch commands.
    """
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, 'csv_files')
    output_dir = os.path.join(script_dir, OUTPUT_DIR)

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_scripts = []
    all_sbatch_commands = []

    # List of CSV files to process
    csv_files = [
        # Two-omics file
        'MLTasks_TwoFeaturesCombinations_Year02(Sheet1).csv',
        # Three-omics files
        'black_three_omics.csv',
        'asian_three_omics.csv',
        'nat_a_three_omics.csv',
        # Four-omics files
        'black_four_omics.csv',
        'asian_four_omics.csv',
        'nat_a_four_omics.csv',
    ]

    for csv_file in csv_files:
        filepath = os.path.join(csv_dir, csv_file)

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        # Determine file type and process accordingly
        if 'TwoFeatures' in csv_file:
            scripts, commands = process_two_omics_file(filepath, output_dir)
        else:
            scripts, commands = process_three_four_omics_file(filepath, output_dir)

        all_scripts.extend(scripts)
        all_sbatch_commands.extend(commands)

    # Write all scripts
    print(f"\nWriting {len(all_scripts)} scripts...")
    for script_path, script_content in all_scripts:
        with open(script_path, 'w', newline='\n') as f:
            f.write(script_content)

    # Write sbatch commands file
    sbatch_file_path = os.path.join(output_dir, SBATCH_FILE)
    with open(sbatch_file_path, 'w', newline='\n') as f:
        f.write("#!/bin/bash\n")
        f.write("# Generated sbatch commands for multiomics tasks\n")
        f.write(f"# Total tasks: {len(all_sbatch_commands)}\n\n")
        for cmd in all_sbatch_commands:
            f.write(cmd + '\n')

    print(f"\nGeneration complete!")
    print(f"  - Scripts written: {len(all_scripts)}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Sbatch commands file: {sbatch_file_path}")
    print(f"\nTo submit all jobs, run:")
    print(f"  bash {sbatch_file_path}")


if __name__ == '__main__':
    main()
