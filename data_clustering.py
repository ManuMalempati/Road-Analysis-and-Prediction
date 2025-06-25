"""
clustering_analysis.py

Task 4.5: Data Clustering & Risk Profiling

This script performs K-Means clustering on driver demographics and injury severity
to identify risk-based profiles in Victorian crash data. It uses Jia’s preprocessed
dataset, selects the optimal number of clusters (k) using Silhouette analysis,
and outputs cluster labels and profile summaries. It also visualizes the results
with Silhouette Score plots and PCA-based cluster projection.

All output artifacts are saved into the ./data-clustering-results/ folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing       import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose             import ColumnTransformer
from sklearn.pipeline            import Pipeline
from sklearn.cluster             import KMeans
from sklearn.decomposition       import PCA
from sklearn.metrics             import silhouette_score, normalized_mutual_info_score
from sklearn.impute               import SimpleImputer

# Suppress only sklearn’s runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create output folder for results
output_dir = './data-clustering-results'
os.makedirs(output_dir, exist_ok=True)

def load_data(path):
    """
    Load the preprocessed dataset produced during Task 4.2.
    """
    return pd.read_csv(path)

def select_features(df, cols):
    """
    Select the subset of features to be used in clustering.
    """
    return df[cols].copy()

def pick_top_nmi_features(df, target_col, n=4):
    """
    Compute NMI between each column and the target, then return the top-n features.
    Both series are cast to string to avoid mixed-type sorting errors.
    """
    nmi_scores = {}
    # prepare target labels as strings
    target_series = df[target_col].fillna('NA').astype(str)
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique(dropna=True) > 1:
            feature_series = df[col].fillna('NA').astype(str)
            nmi = normalized_mutual_info_score(feature_series, target_series)
            nmi_scores[col] = nmi

    top_feats = sorted(nmi_scores, key=nmi_scores.get, reverse=True)[:n]
    print(f"Top {n} features by NMI with {target_col}: {top_feats}")
    return top_feats

def build_preprocessor(df, cat_cols, num_cols):
    """
    Build a ColumnTransformer that:
      - imputes missing categoricals with mode + one-hot-encodes (drop='first')
      - imputes missing numerics with median + scales
    Drops any constant columns to avoid zero-variance.
    """
    # drop zero-variance columns
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) > 1]
    num_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1]

    transformers = []
    if cat_cols:
        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('cat', cat_pipeline, cat_cols))

    if num_cols:
        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler())
        ])
        transformers.append(('num', num_pipeline, num_cols))

    return ColumnTransformer(transformers=transformers)

def get_kmeans_pipeline(preprocessor, k):
    """
    Return a Pipeline that:
      1) applies our preprocessor,
      2) cleans out any inf/NaN,
      3) runs KMeans(n_clusters=k).
    """
    clean = FunctionTransformer(
        lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
        validate=False
    )
    return Pipeline([
        ('prep',  preprocessor),
        ('clean', clean),
        ('km',    KMeans(n_clusters=k,
                         random_state=42,
                         n_init=5,
                         max_iter=200))
    ])

def find_optimal_k(X, preprocessor, k_min=2, k_max=8):
    """
    Identify the optimal number of clusters using Silhouette scores
    over k_min…k_max (inclusive). Samples at most 5k points for speed.
    """
    ks     = list(range(k_min, k_max + 1))
    scores = []

    for k in ks:
        pipe   = get_kmeans_pipeline(preprocessor, k)
        labels = pipe.fit_predict(X)

        X_clean = pipe.named_steps['clean'].transform(
            pipe.named_steps['prep'].transform(X)
        )

        sample_size = min(len(X_clean), 5000)
        score = silhouette_score(
            X_clean, labels,
            sample_size=sample_size,
            random_state=42
        )
        scores.append(score)
        print(f"k={k}, silhouette={score:.4f}")

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 4))
    plt.plot(ks, scores, marker='o')
    plt.title('Silhouette Score vs k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'))
    plt.close()

    best_k = ks[scores.index(max(scores))]
    print(f"\nOptimal k = {best_k}\n")
    return best_k

def assign_clusters(X, preprocessor, k):
    """
    Fit KMeans with the chosen k and return:
      - a pd.Series of cluster labels
      - the cleaned, transformed matrix (for PCA & plotting).
    """
    pipe   = get_kmeans_pipeline(preprocessor, k)
    labels = pipe.fit_predict(X)

    X_clean = pipe.named_steps['clean'].transform(
        pipe.named_steps['prep'].transform(X)
    )
    return pd.Series(labels, name='cluster'), X_clean

def profile_clusters(df, features, labels):
    """
    Summarize each cluster by taking the mode of each feature.
    """
    df2 = df.copy()
    df2['cluster'] = labels

    def safe_mode(col):
        return col.mode().iloc[0] if not col.mode().empty else np.nan

    return (
        df2
        .groupby('cluster')[features]
        .agg(safe_mode)
        .reset_index()
    )

def plot_pca(X_transformed, labels):
    """
    Reduce to 2D with PCA & scatterplot clusters.
    """
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_transformed)
    pca_df = pd.DataFrame(coords, columns=['PC1','PC2'])
    pca_df['cluster'] = labels

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='tab10')
    plt.title('K-Means Clustering Visualized with PCA')
    plt.savefig(os.path.join(output_dir, 'kmeans_cluster_pca.png'))
    plt.close()
    print("PCA plot saved successfully.")

def data_clustering():
    """
    1) Load data
    2) Pick top-4 NMI features vs INJ_LEVEL
    3) Build preprocessor
    4) Find optimal k
    5) Assign clusters
    6) Profile & print
    7) Save outputs
    8) Plot PCA
    """
    # 1) Load
    df = load_data('./processed/full_preprocessed.csv')
    print("Data loaded:", df.shape)

    # 2) Top 4 features by NMI against injury level
    target_col = 'INJ_LEVEL'
    top4 = pick_top_nmi_features(df, target_col, n=4)

    # Prepare X
    X = select_features(df, top4 + [target_col])
    cat_cols = [c for c in top4 if X[c].dtype == 'object' or X[c].dtype.name == 'category']
    num_cols = [c for c in top4 if c not in cat_cols]

    # 3) Preprocessor
    preproc = build_preprocessor(X, cat_cols, num_cols)

    # 4) Find best k
    print("\nFinding optimal k…")
    best_k = find_optimal_k(X[top4], preproc, k_min=2, k_max=8)

    # 5) Assign clusters
    labels, X_clean = assign_clusters(X[top4], preproc, best_k)
    df['cluster'] = labels

    # 6) Profile clusters
    summary = profile_clusters(df, top4, labels)
    print("\nCluster Profiles:\n", summary.to_string(index=False))

    # 7) Save labels and summary
    df[['ACCIDENT_NO','PERSON_ID','cluster']] \
      .to_csv(os.path.join(output_dir,'driver_cluster_labels.csv'), index=False)
    summary.to_csv(os.path.join(output_dir,'cluster_profile_summary.csv'), index=False)
    print(f"\nSaved outputs to {output_dir}")

    # 8) PCA visualization
    plot_pca(X_clean, labels)
    print("Data clustering completed successfully.")
