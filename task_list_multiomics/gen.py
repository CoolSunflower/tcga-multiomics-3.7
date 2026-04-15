# -*- coding: utf-8 -*-
"""
Generate tasks.json - master list of all task configurations.
Then use SLURM array jobs to run tasks by index.
"""

import os
import json
import pandas as pd
import re


def parse_feature_type(feature_str):
    """Parse "('Protein', 'mRNA')" to "Protein_mRNA" """
    features = re.findall(r"'([^']+)'", feature_str)
    return '_'.join(features) if features else feature_str


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, 'csv_files')

    # Feature method configurations: (FeatureMethod, AutoencoderSettings, name)
    feature_methods = [
        (0, None, "ANOVA"),
        (1, None, "PCA"),
        (2, 1, "AE-1"),
        (2, 2, "AE-2"),
    ]
    features_counts = [100, 200]

    all_tasks = []
    task_id = 0

    # Process all CSV files
    csv_files = [
        ('MLTasks_TwoFeaturesCombinations_Year02(Sheet1).csv', None),  # DDP from column
        ('black_three_omics.csv', 'BLACK'),
        ('asian_three_omics.csv', 'ASIAN'),
        ('nat_a_three_omics.csv', 'NAT_A'),
        ('black_four_omics.csv', 'BLACK'),
        ('asian_four_omics.csv', 'ASIAN'),
        ('nat_a_four_omics.csv', 'NAT_A'),
    ]

    for csv_file, ddp_from_filename in csv_files:
        filepath = os.path.join(csv_dir, csv_file)
        if not os.path.exists(filepath):
            print(f"Skip: {csv_file} not found")
            continue

        df = pd.read_csv(filepath)
        print(f"Processing {csv_file}: {len(df)} rows")

        for _, row in df.iterrows():
            # Extract base task info based on file type
            if ddp_from_filename is None:  # Two-omics file
                cancer_type = row['Cancer Type']
                omics_feature = parse_feature_type(row['Feature Type'])
                endpoint = row['Clinical Outcome Endpoint']
                years = int(row['Event Time Threshold (Year)'])
                ddp_group = row['Target Group'].upper()
            else:  # Three/four-omics files
                cancer_type = row['Cancer_type']
                omics_feature = parse_feature_type(row['Feature_type'])
                endpoint = row['Target']
                years = int(row['Years'])
                ddp_group = ddp_from_filename

            # Expand with all feature method + count combinations
            for fc in features_counts:
                for fm, ae_setting, method_name in feature_methods:
                    all_tasks.append({
                        "id": task_id,
                        "data_Category": "Race",
                        "DDP_group": ddp_group,
                        "cancer_type": cancer_type,
                        "omics_feature": omics_feature,
                        "endpoint": endpoint,
                        "years": years,
                        "features_count": fc,
                        "FeatureMethod": fm,
                        "AutoencoderSettings": ae_setting,
                    })
                    task_id += 1

    # Save tasks.json
    tasks_file = os.path.join(script_dir, 'tasks.json')
    with open(tasks_file, 'w') as f:
        json.dump(all_tasks, f, indent=2)

    print(f"\nGenerated {len(all_tasks)} tasks -> tasks.json")
    print(f"\nUsage:")
    print(f"  Single task:  python main.py --task_file task_list_multiomics/tasks.json --task_index 0")
    print(f"  Range:        python main.py --task_file task_list_multiomics/tasks.json --start_index 0 --end_index 100")
    print(f"  SLURM array:  sbatch --array=0-{len(all_tasks)-1}%50 run_array.sh")


if __name__ == '__main__':
    main()
