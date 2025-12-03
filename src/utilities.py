"""The utilities module provides common utilities used throughout the Capstone project.

This project contains common functions that are vital to the overall success of the Capstone project. Rather than duplicating code, this sub-module was created in order to smooth over the development process, and provide standardization for common tasks.

The functions in this module include creating a uniform logger, robustly loading data from the FRED API, and committing data files to Weights&Biases as needed.
"""
# Imports
import logging
import os
from pathlib import Path
import pandas as pd
import sys


def new_logger(logger_name: str, rel_dir_path: str, log_level=logging.DEBUG) -> logging.Logger:
    """Standardizes logs across the project for easier troubleshooting.

    The project logger utilizes two handlers: a FileHandler and a StreamHandler.

    Args:
        logger_name (str):
            The part of the program being logged. Required.
        rel_dir_path (str):
            The relative path to the logging directory from that part of the program. Required.
        log_level (int):
            The minimum level of logging to use. Will take one of the following values:
                logging.DEBUG (10)
                logging.INFO (20)
                logging.WARNING (30)
                logging.ERROR (40)
                logging.CRITICAL (50)

    Returns:
        An object of type `logging.Logger` that is fully configured for the part of the program from which it was called.
    """
    logging.captureWarnings(True)
    # basic logger object, uses the required parameter logger_name to differentiate in the logs
    logger = logging.getLogger(logger_name)

    # a list of dictionary objects in form "level" and "message". Will be used to log any pre-logging setup warnings for later use.
    pre_log_messages = []

    if log_level not in [x * 10 for x in range(1,6)]:
        pre_log_messages.append(
            {
                "level": logging.WARNING,
                "message": f"Unable to use value of '{log_level}' for logging. Used {logging.DEBUG} instead."
            }
        )
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level)

    # Creating Handlers
    # check to make sure directory exists for the rotating file log
    os.makedirs(Path(rel_dir_path), exist_ok=True)

    fh = logging.FileHandler(f'{rel_dir_path}/{logger_name}.log',mode='a',encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    # stream being the console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)

    # Creating Formatter
    # common formatter for all logs in project
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z")
    # add formatter to both handlers
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    # add both handlers to main logger, if they don't already exist
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    if len(pre_log_messages) > 0:
        for message in pre_log_messages:
            logger.log(level=message["level"], msg=message["message"])

    return logger

# Utilities Module-Wide Logging
util_logger = new_logger(__name__, 'logs/utils')

def save_atomic(df: pd.DataFrame, data_path: Path, fmt: str = "parquet") -> Path:
    """Implements an atomic save design pattern that will prevent users from seeing partially written files.

    Performs this using the OS-specific .replace() function on a temporary file that will fully overwrite the old file, without leaving it partially completed for users who open the file in the middle of the write operation.

    The default for writing the cache file to disk is Parquet (https://parquet.apache.org/docs/file-format/). This is because this file format preserves type information while saving space. It can save space because it is a binary file format that is able to be efficiently compressed.

    Args:
        df (pd.DataFrame):
            The Pandas DataFrame to write to the local file system.
        data_path (Path):
            The original path of the data file to overwrite.
        fmt (str):
            The format to use to write the file to local disk. Defaults to 'parquet'.

    Returns:
        Path, the data path for logging in artifact trackers.
    """

    # create a temporary file that will replace the cached file
    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    util_logger.debug(f"Created temporary file {tmp.name}")
    # save in various formats depending on the supplied format
    match fmt:
        case "parquet":
            # preserves type information, not the index
            df.to_parquet(tmp, index=True)
        case "feather":
            # does not preserve type information, smaller file format for most simple use cases
            df.to_feather(tmp)
        case _:
            # does not preserve type information, plain text file format for simple use cases
            df.to_csv(tmp, index=True)
    
    util_logger.info(f"Saved content to {tmp.name} successfully, performing atomic swap.")
    
    # swap tmp with data_path using replace()
    tmp.replace(data_path)
    util_logger.info(f"{data_path.name} is now the new version.")

    return data_path