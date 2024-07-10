import pandas as pd
import numpy as np
from typing import List

def preprocess(df: pd.DataFrame, file_name: str, content_column: List[str], dataset_iteration: int, n: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    if content_column:
        data_name = content_column[dataset_iteration]
        if data_name not in df.columns:
            raise ValueError(f"{data_name} is not a column in {file_name}")
        
        df = df[[data_name] + [col for col in df.columns if col != data_name]]
    else:
        data_name = df.select_dtypes(include=['object']).apply(lambda x: x.str.len().sum()).idxmax()
        df = df[[data_name] + [col for col in df.columns if col != data_name]]
    
    max_rows = len(df)
    rows = np.random.choice(df.index, size=min(n, max_rows), replace=False)
    df = df.loc[rows].reset_index(drop=True)
    df[df.columns[0]] = df[df.columns[0]].str.strip()
    
    return df, rows, max_rows

def preprocess_duplicates(df: pd.DataFrame, title_column: str, method: str = 'suffix') -> pd.DataFrame:
    """
    Preprocess the DataFrame to handle duplicate values in the title column.
    
    :param df: Input DataFrame
    :param title_column: Name of the column to check for duplicates
    :param method: Method to handle duplicates. Options: 'suffix', 'remove', 'concatenate'
    :return: Preprocessed DataFrame
    """
    if method == 'suffix':
        # Current behavior: add suffixes to duplicates
        if df[title_column].duplicated().any():
            df[title_column] = df[title_column] + '_' + df.groupby(title_column).cumcount().astype(str)
    elif method == 'remove':
        # Remove duplicate entries
        df = df.drop_duplicates(subset=[title_column])
    elif method == 'concatenate':
        # Concatenate with another column (e.g., 'ID') to make unique
        if 'ID' in df.columns:
            df[title_column] = df[title_column] + '_' + df['ID'].astype(str)
        else:
            raise ValueError("'ID' column not found for concatenation method")
    else:
        raise ValueError("Invalid method. Choose 'suffix', 'remove', or 'concatenate'")
    
    return df

def data_frame_init(file_name: str, content_column: List[str], dataset_iteration: int, n: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    df = pd.read_csv(file_name)
    return preprocess(df, file_name, content_column, dataset_iteration, n)

