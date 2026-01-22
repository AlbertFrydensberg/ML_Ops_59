from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_ops_59.artifacts import load_artifacts

app = FastAPI(title="ml_ops_59 wine classifier", version="0.1.0")

# Loaded on startup
MODEL = None
SCALER = None
FEATURE_NAMES: List[str] = []


class PredictRequest(BaseModel):
    # Accept either:
    #  - a list of floats in the exact training feature order, OR
    #  - a dict mapping feature_name -> value (order irrelevant)
    x: Union[List[float], Dict[str, float]] = Field(
        ...,
        description="Either a feature list in training order OR a {feature_name: value} mapping.",
    )


class PredictBatchRequest(BaseModel):
    xs: List[Union[List[float], Dict[str, float]]] = Field(
        ...,
        description="Batch of inputs (each either list or dict).",
    )


@app.on_event("startup")
def _load() -> None:
    global MODEL, SCALER, FEATURE_NAMES
    MODEL, SCALER, FEATURE_NAMES = load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }


def _to_row(x: Union[List[float], Dict[str, float]]) -> np.ndarray:
    if isinstance(x, list):
        if len(x) != len(FEATURE_NAMES):
            raise HTTPException(
                status_code=422,
                detail=f"Expected list of length {len(FEATURE_NAMES)}, got {len(x)}.",
            )
        return np.array(x, dtype=float)

    # dict case
    missing = [f for f in FEATURE_NAMES if f not in x]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}",
        )

    # (optional) reject extras to avoid silent mistakes
    extras = [k for k in x.keys() if k not in FEATURE_NAMES]
    if extras:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown features: {extras}",
        )

    return np.array([x[f] for f in FEATURE_NAMES], dtype=float)


@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    row = _to_row(req.x).reshape(1, -1)
    row_scaled = SCALER.transform(row)
    pred = MODEL.predict(row_scaled)[0]

    # dataset target is 1-3 in many wine variants; return as string per requirement
    return {"predicted_class": str(pred)}


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    rows = np.vstack([_to_row(x) for x in req.xs])
    rows_scaled = SCALER.transform(rows)
    preds = MODEL.predict(rows_scaled)
    return {"predicted_classes": [str(p) for p in preds]}
