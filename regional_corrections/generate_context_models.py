#!/usr/bin/env python3
"""
Script to generate logistic regression models and PCA transformations 
for all trinucleotide contexts based on genomic features.

This script recreates the missing pickle files that are used in 
run_nc_constraint_gnomad_v31_main.py for mutation rate adjustment.

Usage:
    python generate_context_models.py

This script follows the same GCP bucket structure as analyze_individual_feature_effects.py
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
from sklearn.decomposition import IncrementalPCA
import pickle
import os
import sys
import csv

# GCP bucket and directory configuration (matching analyze_individual_feature_effects.py)
input_bucket = 'gs://panchal-sandbox-storage/non_coding_constraint'
output_prefix = "genomic_features_cocktail"
output_dir = '/home/rpanchal'

def setup_directories():
    """Set up required directories and download data from GCS"""
    print("Setting up directories and downloading data from GCS...")
    
    # Create genomic_features_syn directory if it doesn't exist
    if not os.path.exists('{0}/{1}'.format(output_dir,output_prefix)): 
        os.mkdir('{0}/{1}'.format(output_dir,output_prefix))
    
    # Create logit_models subdirectory
    if not os.path.exists('{0}/{1}/logit_models'.format(output_dir,output_prefix)):
        os.mkdir('{0}/{1}/logit_models'.format(output_dir,output_prefix))
    
    # Download all required files from GCS
    print("Downloading data files from GCS bucket...")
    os.system('gsutil cp {0}/{1}/* {2}/{1}'.format(input_bucket,output_prefix,output_dir))
    
    print("Data download completed.")

def check_required_files():
    """Check if all required training data files are present"""
    print("Checking for required training data files...")
    
    required_files = [
        '{0}/{1}/cocktail_DNMs_by_context_methyl.txt'.format(output_dir,output_prefix),
        '{0}/{1}/genomic_features13_dnm1_flnk_1k-1M.txt'.format(output_dir,output_prefix),
        '{0}/{1}/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt'.format(output_dir,output_prefix),
        '{0}/{1}/genomic_features13_dnm0_10x_flnk_1k-1M.txt'.format(output_dir,output_prefix),
        '{0}/{1}/mutation_rate_by_context_methyl.txt'.format(output_dir,output_prefix),
        '{0}/{1}/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.selected.txt'.format(output_dir,output_prefix)
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all required training data files are in the GCS bucket:")
        print(f"  {input_bucket}/{output_prefix}/")
        return False
    
    print("All required files found.")
    return True

def load_feature_data():
    """Load and prepare feature data for training"""
    print("Loading feature selection data...")
    
    # Load the selected features by context (output from analyze_individual_feature_effects.py)
    df_ft_sel = pd.read_csv('{0}/{1}/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.selected.txt'.format(output_dir,output_prefix), sep='\t')
    print(f"Loaded {len(df_ft_sel)} selected features across contexts")
    
    return df_ft_sel

def get_contexts():
    """Get all unique contexts from the mutation rate file"""
    dnm_po = pd.read_csv('{0}/{1}/mutation_rate_by_context_methyl.txt'.format(output_dir,output_prefix), sep='\t')
    contexts = sorted(list(set(dnm_po['context'])))
    print(f"Found {len(contexts)} contexts: {contexts}")
    return contexts

def load_training_data():
    """Load training data for model fitting"""
    print("Loading training data...")
    
    # Load de novo variants (following analyze_individual_feature_effects.py pattern)
    df_dnm1 = pd.read_csv('{0}/{1}/cocktail_DNMs_by_context_methyl.txt'.format(output_dir,output_prefix), sep='\t')
    df_dnm1 = df_dnm1[~df_dnm1['locus'].str.contains('chrX:')]
    df_ft_1 = pd.read_csv('{0}/{1}/genomic_features13_dnm1_flnk_1k-1M.txt'.format(output_dir,output_prefix), sep='\t').drop_duplicates()
    df_dnm1 = df_dnm1.merge(df_ft_1.rename(columns={'element_id':'locus'}), how='left', on='locus')
    
    # Load 'non-mutated' background
    df_dnm0 = pd.read_csv('{0}/{1}/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt'.format(output_dir,output_prefix), sep='\t')
    df_dnm0 = df_dnm0[~df_dnm0['locus'].str.contains('chrX:')]
    df_ft_0 = pd.read_csv('{0}/{1}/genomic_features13_dnm0_10x_flnk_1k-1M.txt'.format(output_dir,output_prefix), sep='\t').drop_duplicates()
    df_dnm0 = df_dnm0.merge(df_ft_0.rename(columns={'element_id':'locus'}), how='left', on='locus')
    
    print(f"Loaded training data: {len(df_dnm1)} mutated sites, {len(df_dnm0)} background sites")
    
    return df_dnm1, df_dnm0

def prepare_features_for_context(context, df_ft_sel):
    """
    Prepare feature matrix for a specific context based on selected features.
    
    Args:
        context: Trinucleotide context (e.g., 'AAA')
        df_ft_sel: DataFrame with selected features by context
    
    Returns:
        feature_names: List of selected feature names for this context
    """
    # Get selected features for this context
    context_features = df_ft_sel[df_ft_sel['context'] == context]
    
    # Features correlated with methylation (excluded for CpG contexts)
    ft_corr_met = ['GC_content', 'SINE', 'met_sperm', 'Nucleosome', 'CpG_island']
    
    # Exclude methylation-correlated features for CpG contexts
    if context in ['ACG', 'CCG', 'GCG', 'TCG']:
        context_features = context_features[~context_features['feature'].isin(ft_corr_met)]
    
    # Build feature column names (feature + window)
    feature_names = list(context_features['feature'] + '_' + context_features['window'])
    
    print(f"Context {context}: {len(feature_names)} features selected")
    
    return feature_names

def prepare_training_data(context, df_dnm1, df_dnm0, feature_names):
    """
    Prepare training data for a specific context.
    
    Args:
        context: Trinucleotide context
        df_dnm1: Mutated sites data
        df_dnm0: Background sites data
        feature_names: List of feature names to use
    
    Returns:
        X: Feature matrix
        y: Target vector (1 for mutated, 0 for background)
    """
    print(f"Preparing training data for context {context}...")
    
    # Filter data for this context
    df_1_ = df_dnm1[df_dnm1['context'] == context]
    df_0_ = df_dnm0[df_dnm0['context'] == context]
    
    # Prepare feature matrices
    df_1 = df_1_[['locus'] + feature_names].dropna()
    df_1['group'] = 1
    df_0 = df_0_[['locus'] + feature_names].dropna()
    df_0['group'] = 0
    
    # Combine datasets
    df_01 = pd.concat([df_1, df_0])
    
    # Extract features and target
    X = df_01[feature_names]
    y = df_01['group']
    
    print(f"Training data prepared: {len(df_1)} mutated, {len(df_0)} background sites")
    
    return X, y

def standardize_features(X, save_stats=True, context=None):
    """
    Standardize features using z-score normalization.
    
    Args:
        X: Feature matrix
        save_stats: Whether to save mean/std statistics
        context: Context name for saving stats
    
    Returns:
        X_scaled: Standardized feature matrix
        feature_stats: Dict with means and stds
    """
    print("Standardizing features...")
    
    # Apply z-score normalization (following analyze_individual_feature_effects.py)
    X_scaled = X.apply(scipy.stats.zscore)
    
    # Store feature statistics
    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            'mean': X[col].mean(),
            'std': X[col].std()
        }
    
    if save_stats and context:
        # Save feature statistics in the same format as the example
        stats_file = '{0}/{1}/logit_models/logit_regularized_dnm01_{2}_pbonf_pca.ft_mean_std.txt'.format(output_dir,output_prefix, context)
        with open(stats_file, 'w') as f:
            for feature, stats in feature_stats.items():
                f.write(f"{feature}\t{stats['mean']}\t{stats['std']}\n")
        print(f"Saved feature statistics to {stats_file}")
    
    return X_scaled, feature_stats

def fit_pca_model(X, n_features):
    """
    Fit PCA model for dimensionality reduction.
    
    Args:
        X: Standardized feature matrix
        n_features: Number of pre-selected features (determines PCA components)
    
    Returns:
        pca: Fitted PCA model
        X_pca: PCA-transformed features
    """
    # Number of PCA components should match the number of pre-selected features
    n_components = n_features
    
    print(f"Fitting PCA model with {n_components} components (matching {n_features} pre-selected features)...")
    
    # Use IncrementalPCA to match the existing model structure
    pca = IncrementalPCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA model fitted:")
    print(f"  - Components: {pca.n_components_}")
    print(f"  - Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  - Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca, X_pca

def fit_logistic_model(X_pca, y):
    """
    Fit regularized logistic regression model.
    
    Args:
        X_pca: PCA-transformed features
        y: Binary target variable (1 for mutated, 0 for background)
    
    Returns:
        logit_model: Fitted logistic regression model
    """
    print("Fitting regularized logistic regression...")
    
    # Add constant term for intercept
    X_with_const = sm.add_constant(X_pca, has_constant='add')
    
    # Fit regularized logistic regression (L1 regularization)
    try:
        logit_model = sm.Logit(y, X_with_const).fit_regularized(method='l1', alpha=0.0)
        print(f"Logistic regression fitted successfully:")
        print(f"  - Parameters: {logit_model.params}")
        print(f"  - Converged: {logit_model.mle_retvals['converged']}")
        return logit_model
    except Exception as e:
        print(f"Error fitting logistic regression: {e}")
        return None

def save_models(context, pca, logit_model):
    """
    Save PCA and logistic regression models as pickle files.
    
    Args:
        context: Context name (e.g., 'AAA')
        pca: Fitted PCA model
        logit_model: Fitted logistic regression model
    """
    print(f"Saving models for context {context}...")
    
    # Save PCA model
    pca_file = '{0}/{1}/logit_models/logit_regularized_dnm01_{2}_pbonf_pca.pca.pkl'.format(output_dir,output_prefix, context)
    with open(pca_file, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {pca_file}")
    
    # Save logistic regression model
    logit_file = '{0}/{1}/logit_models/logit_regularized_dnm01_{2}_pbonf_pca.pkl'.format(output_dir,output_prefix, context)
    with open(logit_file, 'wb') as f:
        pickle.dump(logit_model, f)
    print(f"Saved logistic model to {logit_file}")

def upload_models_to_gcs():
    """Upload generated model files to GCS bucket"""
    print("Uploading model files to GCS...")
    
    # Upload all generated pickle files and stats files
    os.system('gsutil cp {0}/{1}/logit_models/logit_regularized_dnm01_*_pbonf_pca.pkl {2}/{1}/logit_models/'.format(output_dir,output_prefix, input_bucket))
    os.system('gsutil cp {0}/{1}/logit_models/logit_regularized_dnm01_*_pbonf_pca.pca.pkl {2}/{1}/logit_models/'.format(output_dir,output_prefix, input_bucket))
    os.system('gsutil cp {0}/{1}/logit_models/logit_regularized_dnm01_*_pbonf_pca.ft_mean_std.txt {2}/{1}/logit_models/'.format(output_dir,output_prefix, input_bucket))
    
    print("Model files uploaded to GCS.")

def train_context_model(context, df_ft_sel, df_dnm1, df_dnm0):
    """
    Train models for a specific context.
    
    Args:
        context: Trinucleotide context
        df_ft_sel: Selected features DataFrame
        df_dnm1: Mutated sites data
        df_dnm0: Background sites data
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n=== Training models for context {context} ===")
    
    try:
        # Get feature names for this context
        feature_names = prepare_features_for_context(context, df_ft_sel)
        
        if len(feature_names) == 0:
            print(f"No features selected for context {context}. Skipping.")
            return False
        
        # Prepare training data
        X, y = prepare_training_data(context, df_dnm1, df_dnm0, feature_names)
        
        if len(X) == 0:
            print(f"No training data available for context {context}. Skipping.")
            return False
        
        # Standardize features
        X_scaled, _ = standardize_features(X, save_stats=True, context=context)
        
        # Fit PCA with number of components matching pre-selected features
        pca, X_pca = fit_pca_model(X_scaled, len(feature_names))
        
        # Fit logistic regression
        logit_model = fit_logistic_model(X_pca, y)
        
        if logit_model is None:
            print(f"Failed to fit logistic regression for context {context}")
            return False
        
        # Save models
        save_models(context, pca, logit_model)
        
        print(f"Successfully trained models for context {context}")
        return True
        
    except Exception as e:
        print(f"Error training models for context {context}: {e}")
        return False

def main():
    """Main function to generate models for all contexts"""
    print("=== Generating Context-Specific Models ===")
    print(f"Input bucket: {input_bucket}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Set up directories and download data
    setup_directories()
    
    # Check if all required files are present
    if not check_required_files():
        print("Exiting due to missing files.")
        sys.exit(1)
    
    # Load feature selection data
    df_ft_sel = load_feature_data()
    
    # Get all contexts
    contexts = get_contexts()
    
    # Load training data
    df_dnm1, df_dnm0 = load_training_data()
    
    # Train models for each context
    successful_contexts = []
    failed_contexts = []
    
    for context in contexts:
        success = train_context_model(context, df_ft_sel, df_dnm1, df_dnm0)
        if success:
            successful_contexts.append(context)
        else:
            failed_contexts.append(context)
    
    # Upload models to GCS
    if successful_contexts:
        upload_models_to_gcs()
    
    # Summary
    print("\n=== Training Summary ===")
    print(f"Successfully trained: {len(successful_contexts)} contexts")
    if successful_contexts:
        print(f"  {', '.join(successful_contexts)}")
    
    if failed_contexts:
        print(f"Failed to train: {len(failed_contexts)} contexts")
        print(f"  {', '.join(failed_contexts)}")
    
    print(f"\nModel files saved in: {output_dir}/{output_prefix}/")
    print(f"Model files uploaded to: {input_bucket}/{output_prefix}/")

if __name__ == "__main__":
    main() 