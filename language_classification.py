from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pandas as pd
import numpy as np
from typing import List
def get_classifier(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, device="cuda")

def language_classifier(df: pd.DataFrame, rows: np.ndarray, columns: List[str], file_name: str) -> None:
    model_name = 'qanastek/51-languages-classifier'
    classifier = get_classifier(model_name)
    df_max = pd.read_csv(file_name)
    max_rows = len(df_max)

    for c in columns:
        c_classified = f"{c}_classified"
        if c not in df.columns:
            raise ValueError(f"{c} is not a column in {file_name}")
        
        if c_classified not in df.columns:
            print(f"Classifying languages from '{c}'")
            data = classify_data(classifier, df[c])
            update_dataframes(df, df_max, c_classified, data, rows, max_rows)
        else:
            update_remaining_rows(df, df_max, c, c_classified, classifier, rows)
        print("Language classification complete")
        df_max.to_csv(file_name, index=False)

def classify_data(classifier: TextClassificationPipeline, data: pd.Series) -> List[str]:
    return [d["label"] for d in classifier(list(data.to_numpy()))]

def update_dataframes(df: pd.DataFrame, df_max: pd.DataFrame, column: str, data: List[str], rows: np.ndarray, max_rows: int) -> None:
    df[column] = data
    data_max = np.full(max_rows, "empty", dtype='U100')
    data_max[rows] = data
    df_max[column] = data_max

def update_remaining_rows(df: pd.DataFrame, df_max: pd.DataFrame, c: str, c_classified: str, classifier: TextClassificationPipeline, rows: np.ndarray) -> None:
    mask = df[c_classified] == "empty"
    rows_remaining = df.index[mask]
    if len(rows_remaining) > 0:
        print(f"{len(rows_remaining)} rows have not been classified before")
        data_remaining = classify_data(classifier, df.loc[rows_remaining, c])
        df.loc[rows_remaining, c_classified] = data_remaining
        df_max.loc[rows[rows_remaining], c_classified] = data_remaining