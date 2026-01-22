import os

import kagglehub
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")
reference_data = pd.read_csv(os.path.join(path, os.listdir(path)[0]))

current_data = pd.read_csv("data/raw/wine.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("report.html")
