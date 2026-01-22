import json
import os
from pathlib import Path

from locust import HttpUser, task, between


def _load_feature_count() -> int:
    """
    Tries to read models/metadata.json so the test always matches current model.
    Falls back to 13 if metadata isn't found.
    """
    repo_root = Path(__file__).resolve().parents[2]
    meta_path = repo_root / "models" / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return len(meta["feature_names"])
    return 13


N_FEATURES = _load_feature_count()


class WineApiUser(HttpUser):
    """
    Minimal load test user that repeatedly calls POST /predict.
    """
    wait_time = between(0.1, 0.5)  # small think time between requests

    @task
    def predict(self):
        # Simple deterministic payload: zeros of the right length.
        payload = {"x": [0.0] * N_FEATURES}

        # Use name="/predict" so stats aggregate nicely even if we add query params later
        with self.client.post("/predict", json=payload, name="/predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                if "predicted_class" not in data:
                    resp.failure(f"Missing predicted_class in response: {data}")
