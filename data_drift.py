import pandas as pd
import kagglehub
import os
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")
reference_data = pd.read_csv(os.path.join(path, os.listdir(path)[0]))

current_data = pd.read_csv("data/raw/wine.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("report.html")
