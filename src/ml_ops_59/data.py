import kagglehub
import pandas as pd
import os

# Download latest version


def data_loader():
    path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")
    df = pd.read_csv(os.path.join(path, os.listdir(path)[0]))
    return df
