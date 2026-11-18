import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from common import CAT_COLS, COLUMNS, NUM_COLS
from config import RUNS

model = joblib.load(RUNS / "final_model.joblib")
threshold = json.loads((RUNS / "final_model.json").read_text())["threshold"]
app = FastAPI(title="Income above 50K")
FEATURES = [c for c in COLUMNS if c != "class"]


@app.post("/predict")
def predict(person: dict):
    unknown = [k for k in person if k not in FEATURES]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown fields: {unknown}")
    row = {c: person.get(c) for c in FEATURES}
    X = pd.DataFrame([row])
    for col in NUM_COLS:
        X[col] = pd.to_numeric(X[col]).fillna(0)
    for col in CAT_COLS:
        X[col] = X[col].astype("string")
    proba = float(model.predict_proba(X)[0, 1])
    return {"probability": proba, "above_50K": bool(proba >= threshold), "threshold": threshold}
