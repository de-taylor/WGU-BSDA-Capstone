"""
src.statistical_analysis contains the functions that prepare the predictions data to perform McNemar's test.
"""

# Python 3 Standard Library
from itertools import combinations

# Data Science Modules
import pandas as pd

# Statistical Analysis
from mlxtend.evaluate import mcnemar_table
from statsmodels.stats.contingency_tables import mcnemar

# optimized further by allowing all models to be treated at once, rather than only using a subset of each pairing, this should reduce the amount of space used in memory
def _create_mcnemar_raw_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """_create_mcnemar_raw_table modifies the output predictions data from model training to optimize for use in creating a contingency table to complete McNemar's test.

    Specifically, the predictions data has three columns (test_indices, y_true, y_pred) that were captured as arrays on each observation. This function uses pd.DataFrame.explode() on temporary intermediate DataFrames to expand these out to be rows of their own, thus allowing easier comparison between models, and much easier calculation of the 2x2 contingency table.

    Args:
        predictions_df (pd.DataFrame):
            The predictions dataset generated from nested_cross_validation by way of train_ml_models (in this notebook).

    Returns:
        pd.DataFrame: Produces a raw data table that is easy to use directly in mlxtend.evaluate.mcnemar_table to create a contingency table.
    """
    # remove columns that are inconsequential to this part of the process, if they exist
    remove_cols = ['fold', 'trial']
    for col_name in remove_cols:
        if col_name not in predictions_df.columns.values:
            remove_cols.remove(col_name)

    # preserving the original DataFrame through copy
    preds_copy = predictions_df.drop(columns=remove_cols).copy() if len(remove_cols) > 0 else predictions_df.copy()
        
    intermediate_dfs = []
    for model_name in preds_copy['model'].unique():
        # filter to a single model type
        model_df = preds_copy[preds_copy['model'] == model_name].copy()

        # explode lists into rows
        model_df = model_df.explode(['test_indices', 'y_true', 'y_pred'])

        # rename y_pred to model name
        model_df = model_df.rename(columns={'y_pred': model_name})

        # drop model column
        model_df = model_df.drop(columns=['model'])
        intermediate_dfs.append(model_df)
    
    # merge on test_indices and y_true
    merged = intermediate_dfs[0]
    for other_df in intermediate_dfs[1:]:
        merged = pd.merge(
            merged,
            other_df.drop(columns=['y_true']),
        on='test_indices'
    )

    return merged

def run_mcnemars_test(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """run_mcnemars_test generates a contingency table internally to then retrieve the stastic and p-value from McNemar's test.

    Uses the statsmodels implementation of McNemar's test. Specifically assumes that the total number of non-matching observations is greater than 25, thus using the chi-squared distribution rather than the binomial distribution. Also uses the Edwards (1948) continuity correction, which is the most common variant of this test.

    Args:
        predictions_df (pd.DataFrame):
            The predictions dataset generated from nested_cross_validation by way of train_ml_models (in this notebook).

    Returns:
        pd.DataFrame: produces a DataFrame that contains the model pairing, chi-squared statistic, and p-value from McNemar's test.
    """   
    final_stats = []

    # first manipulate the predictions_df table into a shape conducive to McNemar's test
    mcnemar_raw = _create_mcnemar_raw_table(predictions_df).sort_values(by='test_indices')

    test_pairings = list(combinations(predictions_df.model.drop_duplicates(), 2))

    # This needs to be run pairwise between each pair of models
    for pairing in test_pairings:
        contingency_table = mcnemar_table(
            y_target=mcnemar_raw['y_true'], # shared target values
            y_model1=mcnemar_raw[pairing[0]], # first model in pairing
            y_model2=mcnemar_raw[pairing[1]] # second model in pairing
        )

        # compute X^2 value and p-value, store in a new statistics table
        # using the continuity correction added by Al Edwards (1948)
        mcnemar_results = mcnemar(contingency_table, correction=True, exact=False)
        
        final_stats.append({
            "modelA": pairing[0],
            "modelB": pairing[1],
            "chi2": mcnemar_results.statistic, # type: ignore this is a sklearn.utils.Bunch
            "p_value": mcnemar_results.pvalue # type: ignore this is a sklearn.utils.Bunch
        })

    stats_results = pd.DataFrame(final_stats)

    # Using Bonferroni correction of p-value for simplicity
    stats_results['alpha'] = 0.05/len(test_pairings)
    # Show whether the result is significant or not
    stats_results['significant'] = stats_results['p_value'] < stats_results['alpha']

    return stats_results