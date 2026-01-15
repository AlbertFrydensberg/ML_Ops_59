import os

import kagglehub
import pandas as pd


def data_loader():
    path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")
    df = pd.read_csv(os.path.join(path, os.listdir(path)[0]))
    return df


def save_raw_csv(out_path="data/raw/wine.csv", overwrite=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} already exists")

    df = data_loader()
    df.to_csv(out_path, index=False)


def validate_data(df):
    if "class" not in df.columns:
        raise ValueError("Missing 'class' column")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
