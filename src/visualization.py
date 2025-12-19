"""
src.visualization implements the functions required for producing the final visualizations from this analysis.
"""

# Python 3 Standard Library
from pathlib import Path

# Data Science Modules
## Data Analytics and Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Deliverable 1.4.3
def plot_scores_bar_chart(results: pd.DataFrame, hue_order: list[str], save_location: Path) -> None:
    """plot_scores_bar_chart plots the mean and standard deviation of each metric for each model, allowing comparison of the metrics by classification model.

    Args:
        results (pd.DataFrame):
            The performance dataset from nested_cross_validation by way of train_ml_models (in this notebook).
        hue_order (list[str]):
            The specific order of the models for display in each bar cluster.
    """
    # clean the results table for more professional appearance in the bar chart
    all_results_clean = results.drop(columns=['trial','fold','hyperparameters'])
    all_results_clean = all_results_clean.melt(id_vars=['model'], value_vars=['accuracy', 'precision', 'fbeta', 'roc_auc'], var_name='metric', value_name='score')

    # Replace values in the metric column with proper casing and verbiage
    all_results_clean['metric'] = all_results_clean['metric'].replace('accuracy', 'Accuracy')
    all_results_clean['metric'] = all_results_clean['metric'].replace('precision', 'Precision')
    all_results_clean['metric'] = all_results_clean['metric'].replace('fbeta', 'F-Beta (0.5)')
    all_results_clean['metric'] = all_results_clean['metric'].replace('roc_auc', 'ROC-AUC')

    # Allow the use of LaTeX formatting in Matplotlib labels
    plt.rcParams['text.usetex']

    plt.figure(figsize=(20,10))
    ax = sns.barplot(
        all_results_clean,
        x='metric',
        y='score',
        hue='model',
        palette="colorblind",
        errorbar="sd",
        estimator='mean',
        legend='full',
        hue_order=hue_order
    )
    plt.title("Comparison of Models by Mean Scores")
    plt.ylim(0.7, 1.0)
    ax.set_ylabel(r'Score ($\mu \pm \sigma$)')
    ax.set_xlabel('Model Metric Name')

    # add column labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type='edge', padding=10) # type: ignore
    
    # save to disk
    plt.savefig(save_location)


def plot_stats_heatmap(stats_table: pd.DataFrame, save_location: Path) -> None:
    """plot_stats_heatmap plots the results from each application of McNemar's test, showing the p-value and whether it was below the Bonferroni corrected significance threshold.

    Args:
        stats_table (pd.DataFrame):
            The resulting table from run_mcnemars_test with each model pairing, the chi-squared statistic, and the p-value.
    """
    # first we require a symmetric matrix of p-values and models
    pivot_table = stats_table.pivot(index="modelA", columns="modelB", values="p_value")

    # takes care of the issue with half the cells being empty, leaves diagonal alone
    filled_pivot = pivot_table.combine_first(pivot_table.T)

    # Then, to show significance I'll want to use a heatmap to demonstrate which pairings were significant, and which were not
    # significance mask, whether the p-value is still significant when corrected
    significant = filled_pivot < stats_table['alpha'].values[0]

    # create annotations
    annot = filled_pivot.round(4).astype(str)
    # swap in a * to show that the value is significant after correction
    annot = annot + significant.replace({True: "*", False: ""})

    # masking the upper right half so we don't get repeats
    mask = np.triu(np.ones_like(filled_pivot, dtype=bool))

    plt.figure(figsize=(8,5))
    sns.heatmap(
        filled_pivot,
        annot=annot,
        cmap=sns.light_palette("seagreen", reverse=True, as_cmap=True),
        cbar_kws={'label':'p-value'},
        fmt="",
        mask=mask,
        center=stats_table['alpha'].values[0]
    )
    plt.title("Pairwise McNemar Test p-values (Bonferroni Corrected)")
    
    # save to disk
    plt.savefig(save_location)