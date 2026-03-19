# -*- coding: utf-8 -*-
"""
PCA Feature Extraction Module

This module handles PCA-based feature extraction for multiomics data,
similar to how autoencoders.py handles AE-based extraction.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from preProcess import get_protein, get_mRNA, get_MicroRNA, \
    get_Methylation, standarize_dataset


def load_or_train_pca(omics_feature, folderISAAC, cancer_type, endpoint,
                      groups, genders, data_Category, features_count,
                      FeatureMethod):
    """
    Load existing PCA models or train new ones for each omics type.

    Args:
        omics_feature: Single omics type (str) or tuple of omics types
        folderISAAC: Base folder path
        cancer_type: Cancer type being analyzed
        endpoint: Clinical endpoint
        groups: Race groups
        genders: Gender groups
        data_Category: Data category (Race, Gender, GenderRace)
        features_count: Number of PCA components to extract
        FeatureMethod: Feature method (should be 1 for PCA)

    Returns:
        Dictionary of PCA models keyed by omics type
    """
    if FeatureMethod != 1:
        return None

    # Create PCA models directory if it doesn't exist
    pca_models_dir = os.path.join(folderISAAC, 'PCA_models')
    if not os.path.exists(pca_models_dir):
        os.makedirs(pca_models_dir)

    pca_models = {}
    scalers = {}

    if isinstance(omics_feature, str):
        omics_feature = [omics_feature]

    for omics in omics_feature:
        PCA_ModelName = 'TCGA-' + omics + '-' + str(features_count)
        pca_file_name = os.path.join(pca_models_dir, 'pca_' + PCA_ModelName + '.joblib')
        scaler_file_name = os.path.join(pca_models_dir, 'scaler_' + PCA_ModelName + '.joblib')

        print(f'PCA_ModelName for {omics}: {PCA_ModelName}')
        print(f'PCA file name for {omics}: {pca_file_name}')

        # Check if PCA model already exists
        if os.path.exists(pca_file_name) and os.path.exists(scaler_file_name):
            print(f"PCA model for {omics} exists. Loading the model.")
            pca_models[omics] = joblib.load(pca_file_name)
            scalers[omics] = joblib.load(scaler_file_name)
        else:
            print(f"PCA model for {omics} does not exist. Training the model.")

            # Load data for training (using FeatureTrain=True like autoencoders)
            if omics == 'mRNA':
                dataset = get_mRNA(cancer_type=cancer_type, endpoint=endpoint,
                                   groups=groups, genders=genders,
                                   FeatureMethod=FeatureMethod, FeatureTrain=True)
            elif omics == 'MicroRNA':
                dataset = get_MicroRNA(cancer_type=cancer_type, endpoint=endpoint,
                                       groups=groups, genders=genders,
                                       FeatureMethod=FeatureMethod, FeatureTrain=True)
            elif omics == 'Protein':
                dataset = get_protein(cancer_type=cancer_type, endpoint=endpoint,
                                      groups=groups, genders=genders,
                                      FeatureMethod=FeatureMethod, FeatureTrain=True)
            elif omics == 'Methylation':
                dataset = get_Methylation(cancer_type=cancer_type, endpoint=endpoint,
                                          groups=groups, genders=genders,
                                          FeatureMethod=FeatureMethod, FeatureTrain=True)

            dataset = standarize_dataset(dataset)
            X = dataset['X']
            print(f"The shape of the dataset for PCA training is: {np.shape(X)}")

            # Check if we have enough features
            if X.shape[1] < features_count:
                print(f"Number of features in {omics} ({X.shape[1]}) is less than the specified features_count ({features_count}). Using all features.")
                pca_models[omics] = None
                scalers[omics] = None
                continue

            # Fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train PCA
            pca = PCA(n_components=features_count)
            pca.fit(X_scaled)

            print(f'Trained PCA for {omics} with {features_count} components')
            print(f'Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}')

            # Save models
            joblib.dump(pca, pca_file_name)
            joblib.dump(scaler, scaler_file_name)
            print(f"Saved PCA and scaler for {omics}.")

            pca_models[omics] = pca
            scalers[omics] = scaler

    return {'pca': pca_models, 'scalers': scalers}


def process_omics_pca(datasets, pca_data, omics_feature):
    """
    Apply PCA transformation to omics datasets.

    Args:
        datasets: Dictionary of datasets keyed by omics type
        pca_data: Dictionary containing 'pca' and 'scalers' dictionaries
        omics_feature: Single omics type (str) or tuple of omics types

    Returns:
        Transformed dataset with PCA-reduced features
    """
    pca_models = pca_data['pca']
    scalers = pca_data['scalers']

    if isinstance(omics_feature, str):  # single omics
        X = datasets[omics_feature]['X']

        if pca_models[omics_feature] is not None:
            # Scale and transform
            X_scaled = scalers[omics_feature].transform(X)
            X_pca = pca_models[omics_feature].transform(X_scaled).astype('float32')
            datasets[omics_feature]['X'] = X_pca
        else:
            print("No PCA transformation applied as model is None due to less features than requested.")

        data_out = datasets[omics_feature]

    else:  # multi omics
        # Step 1: Find common sample IDs across all omics datasets
        sample_ids_sets = [set(datasets[omics]['Samples']) for omics in omics_feature]
        common_sample_ids = list(set.intersection(*sample_ids_sets))

        transformed_dfs = []

        # Step 2: Align datasets, scale, and apply PCA
        for omics in omics_feature:
            # Align datasets based on common sample IDs
            aligned_data = pd.DataFrame(datasets[omics]['X'], index=datasets[omics]['Samples'])
            aligned_data = aligned_data.loc[common_sample_ids]

            # Filter out non-unique rows
            aligned_data = aligned_data.loc[~aligned_data.index.duplicated(keep='first')]

            # Scale and transform with PCA
            if pca_models[omics] is not None:
                X_scaled = scalers[omics].transform(aligned_data.values)
                X_pca = pca_models[omics].transform(X_scaled)
                transformed_df = pd.DataFrame(X_pca, index=aligned_data.index)
            else:
                # No PCA available, use original scaled data
                transformed_df = pd.DataFrame(aligned_data.values, index=aligned_data.index)

            transformed_dfs.append(transformed_df)

        # Step 3: Combine all transformed data
        X_combined = pd.concat(transformed_dfs, axis=1).values.astype('float32')

        # Update common_sample_ids after deduplication
        common_sample_ids = list(transformed_dfs[0].index)

        datasets['Combined'] = {'X': X_combined, 'Samples': common_sample_ids}

        # Step 4: Extract metadata from first omics dataset
        combined_sample_ids = datasets['Combined']['Samples']

        def filter_dataset(dataset, combined_sample_ids):
            sample_indices = np.where(np.isin(dataset['Samples'], combined_sample_ids))[0]
            filtered_dataset = {}
            for key, value in dataset.items():
                if isinstance(value, np.ndarray) and value.shape[0] == len(dataset['Samples']):
                    filtered_dataset[key] = value[sample_indices]
                else:
                    filtered_dataset[key] = value
            return filtered_dataset

        # Filter datasets based on combined_sample_ids
        filtered_datasets = {omics: filter_dataset(datasets[omics], combined_sample_ids) for omics in omics_feature}

        # Create combined dataset
        keys_to_compare = ['C', 'E', 'G', 'R', 'T']
        new_key = "_".join(omics_feature)
        datasets[new_key] = {}
        for key in keys_to_compare + ['Samples']:
            datasets[new_key][key] = filtered_datasets[omics_feature[0]][key]

        datasets[new_key]['X'] = datasets['Combined']['X']
        data_out = datasets[new_key]

    return data_out
