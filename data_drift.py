import os

import kagglehub
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report
from sklearn import datasets

path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")
reference_data = pd.read_csv(os.path.join(path, os.listdir(path)[0]))

current_data = pd.read_csv("wine.csv")


report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference_data, current_data=current_data)
snapshot.save_html("report.html")
