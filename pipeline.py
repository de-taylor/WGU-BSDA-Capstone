"""
This is the analysis pipeline script created from the original analysis.ipynb notebook.

It contains the full flow from data loading to visualization.
"""

# Python 3 Standard Library
import argparse
import os
from pathlib import Path
import re

# if you are re-running this on your system, you'll probably need to change this path to match where you are storing these files
PROJECT_ROOT_PATH = Path(f"{os.environ['USERPROFILE']}\\OneDrive\\Education\\WGU\\Capstone")

# making sure I'm in the right directory for EDA, need to be in root
if not re.match(r'.*Capstone$', os.getcwd()):
    os.chdir(PROJECT_ROOT_PATH)

# Data Science Modules
## Data Analytics and Visualization
import numpy as np
import pandas as pd

## Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Custom modules
from src.utilities import new_logger
from src.cleaning import clean_dataset
from src.modeling import create_column_transformer, train_ml_models
from src.statistical_analysis import run_mcnemars_test
from src.visualization import plot_scores_bar_chart, plot_stats_heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TODO: Fill in the required arguments for tweaking the config

    # Create logger for pipeline
    logger = new_logger("pipeline", "logs")

    # Load Data
    # Fetch local dataset
    data_path = Path('data/Loan_approval_data_2025.csv').resolve()
    logger.info(f"Attempting to fetch data from '{data_path}'")

    loan_appr_df = pd.read_csv(data_path)
    logger.info(f"Successfully created DataFrame {loan_appr_df.shape}")

    # Clean dataset
    loan_appr_clean = clean_dataset(loan_appr_df)

    # Split dataset into prediction and response variables
    # split the data into X and y sets, since cross-validation will produce the index splits
    y = loan_appr_clean.pop('loan_status')
    X = loan_appr_clean

    # Create the Column Transformer preprocessor for all columns
    preproc = create_column_transformer(loan_appr_clean)

    # Create the model definitions to train on
    model_definitions = [
        {
            "name": "Support Vector Machine",
            "pipeline": Pipeline([
                ('preprocessing', preproc), # Preprocessing step
                ('clf', SVC(max_iter=-1,
                            random_state=72925, probability=True)) # Model step
            ]),
            "param_grid": {
                "clf__C": [10**x for x in range(-1,3)], # more reasonable C values
                "clf__kernel": ['rbf', 'poly'],
                "clf__degree": [2,3], # for 'poly' kernel only
                "clf__gamma": ['scale', 'auto'] # important for rbf kernel
            }
        },
        {
            "name": "Logistic Regression",
            "pipeline": Pipeline([
                ('preprocessing', preproc), # Preprocessing step
                ('clf', LogisticRegression(penalty='l2', random_state=72925)) # Model step
            ]),
            "param_grid": {
                # C must be positive, starting with default value and moving up on log scale 4 places
                "clf__C": list(np.logspace(-4,4,4)),
                "clf__solver": ['lbfgs', 'sag', 'saga', 'newton-cholesky'],
                "clf__max_iter": [x for x in range(1000,2001,250)]
            }
        },
        {
            "name": "Gaussian Naive Bayes",
            "pipeline": Pipeline([
                ('preprocessing', preproc), # Preprocessing step
                ('clf', GaussianNB()) # Model step
            ]),
            "param_grid": {
                "clf__var_smoothing": list(np.logspace(0,-9, num=100))
            }
        },
        {
            "name": "Adaptive Boosting",
            "pipeline": Pipeline([
                ('preprocessing', preproc), # Preprocessing step
                ('clf', AdaBoostClassifier(random_state=72925)) # Model step
            ]),
            "param_grid": {
                "clf__n_estimators": [x for x in range(50,250,50)],
                "clf__learning_rate": [10**x for x in [-2,-1,0]]
            }
        },
        {
            "name": "Random Forest",
            "pipeline": Pipeline([
                ('preprocessing', preproc), # Preprocessing step
                ('clf', RandomForestClassifier()) # Model step
            ]),
            "param_grid": {
                'clf__n_estimators': [10**x for x in range(0,4)],
                'clf__max_features': ['sqrt'],
                'clf__max_depth': [x for x in range(1,6)],
                'clf__min_samples_split': [x*2 for x in range(1,6)]
            }
        },
    ]

    # Implement the training, uses src.modeling._nested_cross_validation internally
    all_results, all_predictions = train_ml_models(
        model_definitions,
        X,
        y,
        verbose=3,
        num_trials=1,
        discard_saves=False,
        output_path='data',
        save_path='data/_save'
    )

    # Implements the statistical analysis, uses src.statistical_analysis._create_mcnemar_raw_table internally
    stats_results = run_mcnemars_test(all_predictions)

    # Produce the bar chart visual
    model_order = [
            'Gaussian Naive Bayes',
            'Random Forest',
            'Logistic Regression',
            'Support Vector Machine',
            'Adaptive Boosting'
        ]
    plot_scores_bar_chart(all_results, model_order, Path(os.path.join("figures", "results_bar.png")))

    # Produce the heatmap visual
    plot_stats_heatmap(stats_results, Path(os.path.join("figures", "stats_heatmap.png")))