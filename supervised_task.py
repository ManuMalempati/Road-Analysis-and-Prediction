"""
supervised_learning.py

Task 4.4: Supervised Learning Models for Injury Severity Prediction

This script implements and evaluates supervised machine learning models to predict
injury severity in road accidents. It uses preprocessed data (from Task 4.2), compares
Random Forest and Gradient Boosting classifiers, and visualizes performance.

All confusion matrices and metric comparison graphs are saved in ./supervised_task/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Ensure consistent results
np.random.seed(42)

# Create output directory
output_dir = './supervised_task'
os.makedirs(output_dir, exist_ok=True)

def load_and_prepare():
    """
    Load the full preprocessed dataset and engineer a binary severity class.
    """
    df = pd.read_csv('./processed/full_preprocessed.csv')
    df['SEVERITY_CLASS'] = df['INJ_LEVEL'].apply(lambda x: 1 if x in [1, 2] else 0)

    features = [
        'SEX', 'AGE_GROUP', 'HELMET_BELT_WORN', 'ROAD_USER_TYPE',
        'SEATING_POSITION', 'SPEED_ZONE', 'LIGHT_CONDITION',
        'NO_OF_VEHICLES', 'ROAD_GEOMETRY', 'DAY_OF_WEEK'
    ]
    X = df[features].copy()
    y = df['SEVERITY_CLASS']
    return X, y

def split_data(X, y):
    """Split features and target into train and test sets."""
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def create_pipeline():
    """Create a pipeline for preprocessing categorical and numerical features."""
    categorical_cols = ['SEX', 'AGE_GROUP', 'SEATING_POSITION', 'ROAD_USER_TYPE', 'LIGHT_CONDITION', 'ROAD_GEOMETRY', 'DAY_OF_WEEK']
    numerical_cols = ['SPEED_ZONE', 'NO_OF_VEHICLES']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])
    return preprocessor

def evaluate_model(model, X_test, y_test, name):
    """
    Evaluate a trained model and save confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Minor/None', 'Fatal/Serious'],
                yticklabels=['Minor/None', 'Fatal/Serious'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_{name.lower().replace(" ", "_")}.png')

    return {'name': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def supervised_task():
    """
    Run the complete supervised modelling task:
    - Load and preprocess data
    - Train two classifiers
    - Evaluate each model
    - Save visual performance comparisons
    """
    print('Running supervised learning task...')
    print('Please wait...')

    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = create_pipeline()

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, clf in models.items():
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        result = evaluate_model(pipeline, X_test, y_test, name)
        results.append(result)

    # Comparison plot
    names = [r['name'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        values = [r[metric] for r in results]
        plt.plot(names, values, marker='o', label=metric.capitalize())

    plt.title('Model Comparison Metrics')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png')
    print("\nAll visualizations saved to ./supervised_task")
    print("Supervised learning task complete.")
