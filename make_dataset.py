
import kagglehub, pandas as pd, os

# Download latest version
path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")
df = pd.read_csv(os.path.join(path, os.listdir(path)[0]))
