from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# Glue between training & inference

# This module exists to:
# Save models consistently
# Load models safely


def get_repo_root() -> Path:
    # src/ml_ops_59/artifacts.py -> parents[0]=ml_ops_59, [1]=src, [2]=repo root
    return Path(__file__).resolve().parents[2]


def get_model_dir() -> Path:
    return get_repo_root() / "models"


def save_artifacts(
    model: BaseEstimator,
    scaler: StandardScaler,
    feature_names: List[str],
    out_dir: Path | None = None,
) -> None:
    out_dir = out_dir or get_model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "knn_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    (out_dir / "metadata.json").write_text(json.dumps({"feature_names": feature_names}, indent=2))


def load_artifacts(
    model_dir: Path | None = None,
) -> Tuple[BaseEstimator, StandardScaler, List[str]]:
    model_dir = model_dir or get_model_dir()

    model_path = model_dir / "knn_model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists() or not scaler_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts in {model_dir}. Expected: knn_model.joblib, scaler.joblib, metadata.json"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text())
    feature_names = meta["feature_names"]

    return model, scaler, feature_names
