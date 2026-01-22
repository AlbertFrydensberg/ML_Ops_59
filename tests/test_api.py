import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    from ml_ops_59.api import app

    with TestClient(app) as client:
        r = client.get("/health")

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_endpoint_if_artifacts_exist():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models"

    required = [
        model_dir / "knn_model.joblib",
        model_dir / "scaler.joblib",
        model_dir / "metadata.json",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("Model artifacts not found. Run `uv run python -m ml_ops_59.train` first.")

    meta = json.loads((model_dir / "metadata.json").read_text())
    n_features = len(meta["feature_names"])

    from ml_ops_59.api import app

    with TestClient(app) as client:
        x = [0.0] * n_features
        r = client.post("/predict", json={"x": x})

    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert isinstance(data["predicted_class"], str)
