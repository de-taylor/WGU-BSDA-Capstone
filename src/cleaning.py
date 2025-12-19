"""
src.cleaning contains the cleaning function needed for the original training dataset.
"""

# Python 3 Standard Library
from pathlib import Path

# Data Science Modules
import pandas as pd

# Custom modules
from src.utilities import new_logger, save_atomic

# Create logger for cleaning
logger = new_logger("pipeline", "logs")

def clean_dataset(orig_df: pd.DataFrame) -> pd.DataFrame:
    """clean_dataset implements the cleaning steps needed for the loan approvals training dataset

    Args:
        orig_df (pd.DataFrame): The original dataset from the loan approvals CSV file

    Returns:
        pd.DataFrame: The clean dataset, ready for preprocessing.
    """
    # remove the Customer_ID and payment_to_income_ratio columns
    loan_appr_wip = orig_df.drop(columns=['customer_id', 'payment_to_income_ratio'])

    obj_cols = [col for col in loan_appr_wip.dtypes[loan_appr_wip.dtypes == 'object'].index if col != 'customer_id']
    
    # make each object feature a category feature instead
    for col in obj_cols:
        loan_appr_wip[col] = loan_appr_wip[col].astype('category')

    # save clean dataset as Parquet
    data_path = save_atomic(loan_appr_wip, Path("data/loan_approval_data_2025.clean.parquet"), fmt="parquet")
    logger.info(f"Saved the clean dataset to {data_path} {loan_appr_wip.shape}")

    return loan_appr_wip