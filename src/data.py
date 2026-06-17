import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math
import warnings
import logging
from typing import Optional
from sklearn.model_selection import train_test_split
from src.config import config

def load_raw_data():
    """
    Loads the raw dataset from a zip file in the configured directory.
    Assumes the zip files in 'telco.zip'.
    """
    zip_path = f"{config.RAW_DATA_DIR}/telco.zip"
    df_raw_data = pd.read_csv(zip_path, compression="zip")
    return df_raw_data


def clean_structural_data(
    df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic structural cleaning (Deterministic operations).
    Handles column naming, duplicate  removal, and basic type casting, drop columns, trimming features

    Args:
        df(pd.DataFrame): The imput DataFrame
        
    Returns:
        df(pd.DataFrame): The DataFrame with the Stuctural Cleaning
                          Returns None if the cleaning process fails
    """
    
    df_out = df.copy()
    
    # 1. Standarize column names
    df_out.columns = (
        df_out.columns.str.strip().str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
        )

    # 2. Remove duplicates
    df_out = df_out.drop_duplicates()
    
    # 3. Drop Non-Predictive Columns
    df_out = df_out.drop(columns=[c for c in config.COLUMNS_TO_DROP if c in df_out.columns])
    return df_out
    

def isolate_holdout(
    df: pd.DataFrame,
    holdout_output_path: str,
    holdout_size: float=0.10,
    random_state: int=config.RANDOM_SEED
    ) -> Optional[pd.DataFrame]:
    """
    Splits raw data using stratified sampling
    based on 'TARGET_COLUMN' and securely save the holdout set.
    
    Args:
        df: Raw DataFrame to split
        holdout_output_path (str): File path where the holdout set will be saved. 
        holdout_size (float, optional): Proportion of the dataset to include in the holdout split. Default to 0.10
        random_state (int, optional): Controls the  shuffling applied to the data before splitting.
        Default config.RANDOM_SEED
        
    Returns:
        Optional[pd.DataFrame]: The develoment DataFrame (90%) for processing and cleaning for training.
        Returns None if critical extraction occurs.
        
    Raises:
        FileNotFoundError: If the specified file_path does not exist
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If 'TARGET_COLUMN' is missing from the dataset.
    """
    try:
        logging.info(f"Starting Holdout Spliting")
        # 1. Stratified Split
        # We target 'Churn Label' to maintain class balance in both sets
        df_dev, df_holdout = train_test_split(
            df,
            test_size=holdout_size,
            random_state=random_state,
            stratify=df[config.TARGET_COLUMN_RAW]
        )
        
        # 2. The base directory is created if it does not exist, based on the provided path.
        os.makedirs(os.path.dirname(config.INTERIM_DATA_DIR), exist_ok=True)
        df_holdout.to_csv(config.HOLDOUT_RAW_PATH, index=False)
        
        logging.info(f"The Holdout set ({(holdout_size * 100):.0f}%) locked at: {config.INTERIM_DATA_DIR}")
        logging.info(f"Development Set ({(1 - holdout_size) * 100:.0f}%) ready: {df_dev.shape[0]} rows.")
        
        return df_dev
    
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        logging.error(f"Data validation error during ingestion: {e}")
        return None
    
    
def structural_removal(
    columns_to_drop: list,
    df: pd.DataFrame
    ) -> Optional [pd.DataFrame]:
    """
    Removes a specified list of columns from a DataFrame.
    
    Args:
        columns_to_drop (list): List of column names to be dropped.
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Optional[pd.DataFrame]: The DataFrame without the specified columns.
                                Returns None if the dropping process fails.
    Raises:
        KeyError: If one or more columns in the list are not found in the DataFrame.
    """
    # Preventing DF None
    if df is None:
        logging.error("Critical: Received a  NoneType object instead of a DataFrame. Pipeline aborted")
        return None
    
    try:
        # Inmutability: Work on a copy
        # Execution
        df_out = df.drop(columns=columns_to_drop)
        
        logging.info(f"Successfully dropped {len(columns_to_drop)} columns: {columns_to_drop}")
        logging.info(f"Quantity of Columns before dropping: {df.shape[1]} and after: {df_out.shape[1]}")
        return df_out
     
    except KeyError as e:
        logging.error(f"Failed to drop Columns. Column not found {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during structural removal: {e}")
        return None


def standarize_columns(
    df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
    """
    Standardizes column names to snake_case.
    
    Args:
        df(pd.DataFrame): The input DataFrame
        
    Returns:
        df(pd.DataFrame): The DataFrame with the Standarize Columns
                          Returns None if the cleaning process fails
                          
    Raise:
        Cath all unexpected Memory or Pandas Error
    """
    
    # Defensive Check
    if df is None:
        logging.error(f"Critical: Received NoneType instead of DataFrame")
        return None
    
    try:
        # Inmutability: Work on a copy
        df_out = df.copy()
        
        # Execution()
        df_out.columns = (
            df_out.columns.str.strip().str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        logging.info("Column names succesfully standarized to snake_case")
        return df_out
    
    except Exception as e:
        # Catch-all for unexpected memory or pandas errors
        logging.error(f"Unexpected error during column standarization {e}")
        return None


def trim_string_columns(
    df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Trims leading and trailing whitespace from all string columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Optional[pd.DataFrame]: The DataFrame with trimmed string columns.
                                Returns None if the trimming process fails.
    """
    
    # Defensive Check
    if df is None:
        logging.error(f"Critical: Recived NoneType instead of DataFrame. Trimming String Columns aborted")
        return None
    
    try:
        # Inmutability: Work on a copy
        df_out = df.copy()
        
        # Execution
        # Searching for 'object and 'string' types to do not break Math columns
        string_cols = df_out.select_dtypes(include=['object', 'string']).columns
        
        # Exit if no text columns exist
        if len(string_cols) == 0:
            logging.info(f"No string columns found in DataFrame. Trimming skipped.")
            return df_out
        
        # Defensively Strip Spaces
        for col in string_cols:
            # The lambda explicitly checks if 'x' is a string before stripping.
            # This prevent the function from crashing when it hits a true NaN value or a number
            df_out[col] = df_out[col].apply(lambda x: x.strip() if isinstance(x ,str) else x)
            
        # Logging Traceability
        trimmed_columns_list = string_cols.tolist()
        logging.info(f"Successfully trimmed whitespace in {len(trimmed_columns_list)} text columns: {trimmed_columns_list}")
        
        return df_out
    
    except Exception as e:
        # Catch-all for unexpected memory or Pandas errors
        logging.error(f"Unexpected error during string trimming: {e}")
        return None
    
    
def prepare_target_variable(
    df: pd.DataFrame,
    target_col: str=config.TARGET_COLUMN_CLEAN
) -> Optional[pd.DataFrame]:
    """
    Prepares target variable by dropping NaNs, standardizing text, 
    verifying binary constraints, and mapping to 1/0.

    Args:
        df (pd.Dataframe): The imput DataFrame
        target_col (str, optional): Target. Defaults to TARGET_COLUMN.

    Returns:
        Optional[pd.DataFrame]: Imputed Data Frame with 0 and 1
    """
    if df is None:
        logging.error("Critical: Received NoneType instead of DataFrame")
        return None
    
    if target_col not in df.columns:
        logging.error(f"Critical: Target column '{target_col}' not found in DataFrame")
        return None
    
    try:
        df_out = df.copy()
        
        # Drop rows where target is missing
        initial_rows = len(df_out)
        df_out = df_out.dropna(subset=[target_col])
        if len(df_out) != initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df_out)} rows with missing target values.")
            
        # Standardize text (Forces everything to string for mapping)
        df_out[target_col] = df_out[target_col].astype(str).str.strip().str.lower()
            
        # Fail-Fast Assertion (Unique values check)
        unique_values = df_out[target_col].unique()
        if len(unique_values) != 2:
            raise ValueError(f"CRITICAL: Target must have 2 unique values. Found {len(unique_values)}: {unique_values}")
    
        # Map text to binary
        mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
        
        # We perform the mapping directly since we forced string type in Step 2
        df_out[target_col] = df_out[target_col].map(mapping)
        
        # Final validation of the mapping
        if df_out[target_col].isnull().any():
            unmapped = df_out[df_out[target_col].isnull()][target_col].unique()
            raise ValueError(f"CRITICAL: Mapping failed. Unknown values detected: {unmapped}")
            
        # Final Cast to Integer
        df_out[target_col] = df_out[target_col].astype(int)
            
        logging.info(f"Target variable '{target_col}' successfully prepared.")
        return df_out

    except Exception as e:
        logging.error(f"Pipeline Halted: Unexpected error preparing target variable: {e}")
        raise


def run_data_pipeline() -> bool:
    """
    Orchestrate the phase 1 Data Pipeline.
    Loads data -> Standardizes Names -> Isolate Holdout -> Cleans Dev Data -> Save to Disk.
    """
    logging.info("--- Starting Data Engineering Pipeline (Phase 1) ---")
    
    # 1. Load Data
    df_raw = load_raw_data()
    if df_raw is None:
        return False
    
    # 2. Isolate Holdout
    df_dev = isolate_holdout(df_raw, holdout_output_path=config.HOLDOUT_RAW_PATH)
    if df_dev is None:
        return False
    
    # 3. Deep cleaning Only for develomment set (90%)
    df_dev_cleaned = clean_structural_data(df_dev)
    df_dev_cleaned = trim_string_columns(df_dev_cleaned)
    
    # 4. Prepare Target Variable
    # Explicitly passing the clean target name
    df_dev_cleaned = prepare_target_variable(df_dev_cleaned, target_col=config.TARGET_COLUMN_CLEAN)
    
    if df_dev_cleaned is None:
        logging.error("Failed to clean the development set.")
        return False
    
    # 5. Save the processed data for Notebook 2
    # Using the path from config.py
    df_dev_cleaned.to_csv(config.DEV_CLEANED_PATH, index=False)
    logging.info(f"Phase 1 Completed. Development file saved at: {config.DEV_CLEANED_PATH}")
    return True

if __name__ == "__main__":
    # This block allows the file to be executed directly from the terminal
    success = run_data_pipeline()
    if success:
        logging.info("Pipeline executed successfully.")
    else:
        logging.error("Pipeline failed.")
    