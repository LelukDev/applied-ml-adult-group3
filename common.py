import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import RUNS, ROOT, SEED

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "class"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

NUM_RAW = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
NUM_NEW = ["log_capital_net", "has_capital", "is_married", "overtime_hours", "is_us"]
CAT_ENG = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]
CAT_RAW = CAT_ENG + ["native-country"]


def load_data():
    try:
        df = fetch_openml(name="adult", version=2, as_frame=True).frame
    except Exception:
        df = pd.read_csv(ROOT / "data/raw/adult-all.csv", header=None, names=COLUMNS,
                         skipinitialspace=True, na_values="?")
    df["class"] = df["class"].astype(str).str.strip().str.rstrip(".")
    for col in CAT_COLS:
        df[col] = df[col].astype("string").str.strip().replace({"?": pd.NA})
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col])
    return df


def get_splits():
    df = load_data().drop_duplicates()
    # same rows in the same order whatever the data source (OpenML or csv)
    df = df.loc[df.astype(str).sort_values(list(df.columns)).index].reset_index(drop=True)
    y = (df["class"] == ">50K").astype(int)
    X = df.drop(columns="class")
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def add_features(X):
    X = X.copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("object").where(X[col].notna(), "Missing")
    net = X["capital-gain"] - X["capital-loss"]
    X["log_capital_net"] = np.sign(net) * np.log1p(np.abs(net))
    X["has_capital"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)
    X["is_married"] = X["marital-status"].isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
    X["overtime_hours"] = (X["hours-per-week"] - 40).clip(lower=0)
    X["is_us"] = (X["native-country"] == "United-States").astype(int)
    return X


def make_model(name, class_weight=None, **params):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        args = {"C": 1.0, "max_iter": 2000, **params}
        return LogisticRegression(class_weight=class_weight, **args)
    if name == "random_forest":
        args = {"n_estimators": 400, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": -1, **params}
        return RandomForestClassifier(class_weight=class_weight, random_state=SEED, **args)
    if name == "lightgbm":
        args = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                "verbose": -1, "n_jobs": 4, **params}
        return LGBMClassifier(class_weight=class_weight, random_state=SEED, **args)
    raise ValueError(name)


def build_pipeline(name, class_weight=None, engineered=True, drop=(), smote=False, **params):
    num = NUM_RAW + NUM_NEW if engineered else NUM_RAW
    cat = CAT_ENG if engineered else CAT_RAW
    cat = [c for c in cat if c not in drop]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist",
                              sparse_output=False), cat),
    ], verbose_feature_names_out=False)
    steps = [("features", FunctionTransformer(add_features)),
             ("prep", prep),
             ("model", make_model(name, class_weight, **params))]
    if smote:
        steps.insert(2, ("smote", SMOTE(random_state=SEED)))
        return ImbPipeline(steps)
    return Pipeline(steps)


def load_best():
    return json.loads((RUNS / "best_model.json").read_text())


def tuned_pipeline(**changes):
    best = load_best()
    return build_pipeline(best["model"], **{**best["params"], **changes})


def best_threshold(y, proba):
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def ece_score(y, proba, n_bins=15):
    y, proba = np.asarray(y), np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - proba[mask].mean())
    return float(total)


def get_metrics(y, proba, threshold=0.5):
    y = np.asarray(y)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": float((pred == y).mean()),
        "brier": brier_score_loss(y, proba),
        "ece": ece_score(y, proba),
        "threshold": threshold,
    }


def log_run(stage, model, params, metrics, artifact=""):
    path = RUNS / "run_log.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    row = {"run_id": f"run{len(old) + 1:03d}", "stage": stage, "model": model,
           "params": json.dumps(params, default=str)}
    for key, value in metrics.items():
        row[key] = json.dumps(value) if isinstance(value, list) else value
    row["artifact"] = artifact
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return row["run_id"]
