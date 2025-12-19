"""
src.modeling implements all of the machine learning training steps. This includes creating the column transformer, defining the nested cross validation steps, and finally implementing a robust machine learning training function that performs incremental saves during training.
"""

# Python 3 Standard Library
import os
from pathlib import Path

# Data Science Modules
## Data Analytics and Visualization
import pandas as pd
from tqdm import tqdm

## Machine Learning
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning, UndefinedMetricWarning, NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, fbeta_score

# Custom modules
from src.utilities import new_logger, save_atomic

# Create logger for pipeline
logger = new_logger("modeling", "logs")

def create_column_transformer(clean_df: pd.DataFrame) -> ColumnTransformer:
    """create_column_transformer Uses the clean DataFrame to generate a preprocessor for all major column types in the training dataset.

    Args:
        clean_df (pd.DataFrame): The clean DataFrame output from src.cleaning.clean_dataset

    Returns:
        ColumnTransformer: The ColumnTransformer used to preprocess the data for each machine learning model during training and inference pipelines.
    """
    # Create lists of the columns for different kinds of transformers
    # Categorical columns, currently represented as `object` (string) dtypes
    obj_cols = [col for col in clean_df.dtypes[clean_df.dtypes == 'category'].index]

    # all numerical columns excluding the `loan_status` target variable
    num_cols = clean_df.drop(columns=obj_cols).columns.values

    # Building a column transformer out of OneHotEncoder and StandardScaler
    logger.info("Starting inference training pipeline for all models")
    cat_preproc = OneHotEncoder()
    logger.debug(f"Created OneHotEncoder for columns {obj_cols}")
    num_preproc = StandardScaler()
    logger.debug(f"Created StandardScaler for columns {num_cols}")

    preproc = ColumnTransformer(
        transformers=[
            ("cat_transform", cat_preproc, obj_cols),
            ("num_transform", num_preproc, num_cols)
        ],
        remainder='drop',
        n_jobs=-1
    )
    logger.debug("Created ColumnTransformer for categorical and numerical preprocessing with all above columns. Any other columns will be dropped.")

    return preproc


def _nested_cross_validation(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, est_name: str, param_grid: dict[str, list], trials: int = 1, outer_cv_splits: int = 5, inner_cv_splits: int = 3, random_state: int = 72925, verbose: int = 0, n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Using a Machine Learning Pipeline and Parameter Grid, perform nested Cross-Validation and return a DataFrame with the results.

    Please not that this function can take a LONG time to run, if the hyperparameter space is large, or if the combined number of cross-validation folds and trials is very large.

    This function has a very high time-complexity, and should only be run if you have a lot of time to spare. To put this in common terms, if we have four hyperparameters with 3 options each, we have a hyperparameter space of 

    Args:
        pipeline (sklearn.pipeline.Pipeline):
            The machine learning Pipeline that contains the preprocessed columns and the model to use.
        X (pd.DataFrame):
            The X matrix to use for training and testing, contains only the predictor variables.
        y (pd.Series):
            the y array to use for training and testing, contains only the reponse variable.
        est_name (str):
            The estimator name, for logging.
        param_grid (dict[str, list]):
            A parameter grid dictionary with the parameter names as 'model__parameter' as keys and the list of hyperparameter options as the values.
        trials (int):
            The number of cross-validation trials to perform, defaults to 1.
        outer_cv_splits (int):
            The number of cross-validation splits for each trial, defaults to 5.
        inner_cv_splits (int):
            The number of hyperparameter tuning splits for each outer cross-validation fold, defaults to 3.
        random_state (int):
            The random state to use for better comparison across models. Defaults to 72925.
        verbose (int):
            The level of verbosity to use for the GridSearchCV and cross_validate methods, defaults to 0, letting the loops show the progress.
        n_jobs (int):
            The number of processors to use to parallelize the jobs. Defaults to -1.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            pd.DataFrame: a `results` DataFrame with the trial, fold, model name, F-Beta, Accuracy, Precision, ROC_AUC, and the best hyperparameters for that model.
            
            pd.DataFrame: an `all_preds` DataFrame with the per-observation metrics for each trial/fold for each model. Contains the test indices, the actual test values, and the predicted test values, along with the model name to identify each round correctly.
    """
    logger.info(f"Starting nested cross-validation for {est_name}")
    logger.info(f"Using parameter grid for GridSearchCV: {param_grid}")

    results = [] # aggregate metrics
    all_preds = [] # per-observation predictions for McNemar's test

    try:
        # inner tqdm loop showing trials, from Walters 2022
        for t in tqdm(range(trials), desc=f"{est_name} Cross Validation Trials", leave=True):
            # define Inner CV Loop for hyperparameter tuning
            inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
            
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=30, # number of random samples
                verbose=verbose,
                cv=inner_cv,
                n_jobs=n_jobs,
                random_state=random_state,
                error_score='raise'
            )

            # define Outer CV Loop
            outer_cv = KFold(n_splits=outer_cv_splits, shuffle=True, random_state=random_state)

            for fold_idx, (train_idx, test_idx) in tqdm(enumerate(outer_cv.split(X, y)),total=outer_cv.get_n_splits(),desc=f"Trial {t} - {est_name} Outer Folds",leave=True):
                # fit the model
                random_search.fit(X.iloc[train_idx], y.iloc[train_idx])
                # obtain best estimator from this fitting round
                best_model = random_search.best_estimator_

                # best model's prediction
                y_pred = best_model.predict(X.iloc[test_idx]) # type: ignore
                # the actual results
                y_true = y.iloc[test_idx]

                # store aggregate metrics
                results.append({
                    "trial": t,
                    "fold": fold_idx,
                    "model": est_name,
                    "hyperparameters": random_search.best_params_,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred),
                    "fbeta": fbeta_score(y_true, y_pred, beta=0.5),
                    "roc_auc": roc_auc_score(y_true, best_model.predict_proba(X.iloc[test_idx])[:,1]) # type: ignore
                })

                # store per-observation results
                all_preds.append({
                    "trial": t,
                    "fold": fold_idx,
                    "model": est_name,
                    "test_indices": test_idx,
                    "y_true": y_true.values,
                    "y_pred": y_pred
                })
    except ConvergenceWarning as cw:
        logger.warning(f"The model {est_name} failed to converge.")
        logger.warning(cw.__traceback__)
    except FitFailedWarning as ffw:
        logger.warning(f"The model {est_name} failed to fit.")
        logger.warning(ffw.__traceback__)
    except UndefinedMetricWarning as umw:
        logger.warning(f"An undefined or invalid metric was used.")
        logger.warning(umw.__traceback__)
    except NotFittedError as nfe:
        logger.warning(f"The model {est_name} was not fitted before use.")
        logger.warning(nfe.__traceback__)
    except Exception as err:
        logger.warning(f"The model {est_name} experienced an unknown issue.")
        logger.warning(err.__traceback__)

    return pd.DataFrame(results), pd.DataFrame(all_preds)


def train_ml_models(model_defs: list[dict[str, dict]], X_predictors: pd.DataFrame, y_response: pd.Series, verbose: int = 1, num_trials: int = 1, discard_saves: bool = False, output_path: str = 'data', save_path: str = 'data/_save') -> tuple[pd.DataFrame, pd.DataFrame]:
    """train_ml_models Implements the training steps and incremental backup of results after each training trial for each model.

    Args:
        model_defs (list[dict[str, dict]]):
            This is a dictionary of model definitions that requires the name, pipeline, and param_grid keys. These will be a string, Pipeline() object, and str:list dictionary, respectively.
        X_predictors (pd.DataFrame):
            The observations DataFrame with the predictor variables. The Pipeline() object should contain the required steps to adjust the data before sending it to the classifier.
        y_response (pd.Series):
            The response variable values corresponding to each observation in X by index.
        verbose (int, optional):
            The verbosity level to pass to the _nested_cross_validation function. Defaults to 1.
        num_trials (int, optional):
            The number of trials to perform. Each trial is one full nested cross-validation cycle for one model. Dramatically increases training and fitting time. Defaults to 1.
        discard_saves (bool, optional):
            Whether to clean up all of the incremental saves generated during training. Turn on ONLY if all you want are the final performance and predictions datasets. Defaults to False.
        output_path (str, optional):
            A string denoting the directory relative to the project root where the final Parquet files should be located.. Defaults to 'data'.
        save_path (str, optional): 
            A string denoting the directory relative to the project root where the incremental Parquet files should be located. Defaults to 'data/_save'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The all_results (performance by each metric) and all_predictions (predictions, used for McNemar's test).
    """
    # test for and create the directories
    if not Path(output_path).exists():
        os.mkdir(Path(output_path).resolve())
    if not Path(save_path).exists():
        os.mkdir(Path(save_path).resolve())

    temp_files = [] # locations of checkpoint files for clearing, if needed

    # Saving in the middle of the runs
    temp_results_path = Path(os.path.join(save_path, 'model_metrics.wip.parquet'))
    temp_predictions_path = Path(os.path.join(save_path, 'model_predictions_df.wip.parquet'))

    # Final save at the very end of the run
    final_results_path = Path(os.path.join(output_path, 'model_metrics.final.parquet'))
    final_predictions_path = Path(os.path.join(output_path, 'model_predictions.final.parquet'))

    if not final_results_path.exists() or not final_predictions_path.exists():
        # Starting over, both files must be present in order to read them in
        if not temp_results_path.exists() or not temp_predictions_path.exists():
            # no files found on disk, create an empty DataFrame to start with
            # any existing files will be overwritten in the process
            all_results = pd.DataFrame()
            all_predictions = pd.DataFrame()
        else:
            # load the existing DataFrame and add to it without overwriting initially
            all_results = pd.read_parquet(temp_results_path)
            all_predictions = pd.read_parquet(temp_predictions_path)

        for model in tqdm(model_defs, desc=f"Model Training - Main Experiment Loop", leave=True):
            logger.debug(f"Training iteration with {model['name']} as the inference model.")
            
            _result, _pred = _nested_cross_validation(model['pipeline'], X_predictors, y_response, model['name'], model['param_grid'], trials=num_trials, verbose=verbose) # type: ignore

            logger.debug(f"Results DataFrame for {model['name']} is complete with shape {_result.shape}")

            # log the specific _result DataFrame to disk for recovery purposes
            _model_results_path = Path(os.path.join(save_path, f"model_agg_results_{model['name'].lower().replace(' ', '-')}.wip.parquet")) # type: ignore
            _model_predictions_path = Path(os.path.join(save_path, f"model_preds_{model['name'].lower().replace(' ', '-')}.wip.parquet")) # type: ignore
            
            # save results to disk
            logger.info(f"Saving the results from the {model['name']} training runs to {_model_results_path} for safe-keeping.")
            save_atomic(_result, _model_results_path) # save to disk without resource competition
            temp_files.append(_model_results_path) # track location for cleaning later

            # save predictions to disk
            logger.info(f"Saving the predictions from the {model['name']} training runs to {_model_predictions_path} for safe-keeping.")
            save_atomic(_pred, _model_predictions_path) # save to disk without resource competition
            temp_files.append(_model_predictions_path) # track location for cleaning later

            # add _result to all_results
            if all_results.shape[0] == 0:
                # empty DataFrame
                all_results = _result
                logger.debug(f"Initialized all_results DataFrame from the {model['name']} _result DataFrame.")
            else:
                # concatenating to DataFrame instead
                all_results = pd.concat([all_results, _result])
                logger.debug(f"Added the {model['name']} _result DataFrame to the existing all_results DataFrame. Current shape: {all_results.shape}")
            
            # add _pred to all_predictions
            if all_predictions.shape[0] == 0:
                # empty DataFrame
                all_predictions = _pred
                logger.debug(f"Initialized all_predictions DataFrame from the {model['name']} _pred DataFrame")
            else:
                # concatenating to existing DataFrame
                all_predictions = pd.concat([all_predictions, _pred])
                logger.debug(f"Added the {model['name']} _pred DataFrame to the existing all_predictions DataFrame. Current shape: {all_predictions.shape}")
            
            # log results DataFrame to disk so that it can be recovered on failure
            logger.info(f"Saving the results from all currently completed training runs to {temp_results_path} for safe-keeping.")
            save_atomic(all_results, temp_results_path) # save to disk without resource competition
            # is saved once per model, so make sure not already in list
            if temp_results_path not in temp_files:
                temp_files.append(temp_results_path) # track location for cleaning later

            # log predictions DataFrame to disk so that it can be recovered on failure
            logger.info(f"Saving the predictions from all currently completed training runs to {temp_predictions_path} for safe-keeping.")
            save_atomic(all_predictions, temp_predictions_path) # save to disk without resource competition
            # is saved once per model, so make sure not already in list
            if temp_predictions_path not in temp_files:
                temp_files.append(temp_predictions_path) # track location for cleaning later

        # if you reach this point, CONGRATS you are done, thanks for playing!
        logger.info(f"Saving the final results of training to {final_results_path}.")
        save_atomic(all_results, final_results_path)

        logger.info(f"Saving the final predictions from training to {final_predictions_path}.")
        save_atomic(all_predictions, final_predictions_path)
    else:
        logger.info(f"Loading the final results and predictions from the last full run from disk.")
        # final results have been created, load into DataFrame and don't repeat the process
        all_results = pd.read_parquet(final_results_path)
        all_predictions = pd.read_parquet(final_predictions_path)
    
    if discard_saves:
        # VERIFY That both files are written to disk and then clear out all of the other temp files
        if final_results_path.exists() and final_predictions_path.exists():
            logger.debug("Both the final predictions and final results have been saved. Clearing temporary storage files.")
            for temp_file in temp_files:
                if temp_file != final_results_path and temp_file != final_predictions_path:
                    os.remove(Path(temp_file).resolve())
                    logger.debug(f"Removed temp file {temp_file}")
            
            # remove the temp directory
            os.rmdir(Path(save_path).resolve())
            logger.debug(f"Removed temp directory {save_path}")
    
    # this output is used for the statistical analysis with McNemar's Test
    return all_results, all_predictions