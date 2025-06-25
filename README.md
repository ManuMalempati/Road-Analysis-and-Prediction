# Road Accident Analysis and Prediction

This repository contains a comprehensive analysis of road accident data, including preprocessing, correlation analysis, supervised learning, and clustering tasks. The goal is to identify factors influencing injury severity and risk profiles, and to predict injury severity using machine learning models.

---

## Project Structure

### Directories
- **datasets/**: Contains raw datasets used for analysis.
  - `accident.csv`
  - `atmospheric_cond.csv`
  - `filtered_vehicle.csv`
  - `person.csv`
  - `vehicle.csv`
- **processed/**: Contains the preprocessed dataset.
  - `full_preprocessed.csv`
- **correlation_results/**: Contains results from correlation analysis.
  - `nmi_results.json`
  - `sub_factor_analysis.json`
- **supervised_task/**: Contains outputs from supervised learning models.
  - Confusion matrices and model comparison graphs.
- **data-clustering-results/**: Contains outputs from clustering analysis.
  - `driver_cluster_labels.csv`
  - `cluster_profile_summary.csv`

---

### Files
- **main.py**: Links all tasks together and executes them sequentially.
- **preprocess.py**: Preprocesses the raw datasets, handles missing values, and saves the cleaned dataset.
- **correlation.py**: Performs correlation analysis using Normalized Mutual Information (NMI) and generates charts and JSON outputs.
- **supervised_task.py**: Implements supervised learning models (Random Forest and Gradient Boosting) to predict injury severity.
- **data_clustering.py**: Performs K-Means clustering to identify risk profiles and visualizes clusters using PCA.

---

## Tasks Overview

### 1. Preprocessing
- Merges multiple datasets (`accident.csv`, `person.csv`, `vehicle.csv`, `atmospheric_cond.csv`) into a unified dataset.
- Handles missing values using mode for categorical columns and median for numerical columns.
- Removes outliers using the Interquartile Range (IQR) method.
- Saves the cleaned dataset as `processed/full_preprocessed.csv`.

### 2. Correlation Analysis
- Calculates Normalized Mutual Information (NMI) to identify relationships between factors and injury severity (`SEVERITY`) or injury level (`INJ_LEVEL`).
- Outputs:
  - `nmi_results.json`: NMI scores for all factors.
  - `sub_factor_analysis.json`: Sub-factor analysis for top factors.
  - Bar charts visualizing NMI scores.

### 3. Supervised Learning
- Predicts injury severity using Random Forest and Gradient Boosting classifiers.
- Evaluates models using metrics like accuracy, precision, recall, and F1 score.
- Outputs:
  - Confusion matrices for each model.
  - Comparison graph of model performance.

### 4. Clustering
- Performs K-Means clustering on driver demographics and injury severity.
- Identifies optimal number of clusters using Silhouette analysis.
- Visualizes clusters using PCA.
- Outputs:
  - `driver_cluster_labels.csv`: Cluster labels for each driver.
  - `cluster_profile_summary.csv`: Summary of cluster profiles.

---

## Running the program
- Run the main.py file
