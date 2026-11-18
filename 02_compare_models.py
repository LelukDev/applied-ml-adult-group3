import time

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from common import best_threshold, build_pipeline, get_metrics, get_splits, log_run
from config import MODELS, N_FOLDS, ROOT, SEED, TAB

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}

candidates = [("dummy", "none")]
for name in ["logreg", "random_forest", "lightgbm"]:
    for imbalance in ["none", "balanced", "smote"]:
        candidates.append((name, imbalance))

rows = []
for name, imbalance in candidates:
    start = time.time()
    if imbalance == "smote":
        pipe = build_pipeline(name, smote=True)
    elif imbalance == "balanced":
        pipe = build_pipeline(name, class_weight="balanced")
    else:
        pipe = build_pipeline(name)

    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof)

    pipe.fit(X_train, y_train)
    p_val = pipe.predict_proba(X_val)[:, 1]
    val_05 = get_metrics(y_val, p_val, 0.5)
    val_tuned = get_metrics(y_val, p_val, threshold)

    metrics = {
        "cv_pr_auc_mean": cv_res["test_pr_auc"].mean(),
        "cv_pr_auc_std": cv_res["test_pr_auc"].std(),
        "cv_roc_auc_mean": cv_res["test_roc_auc"].mean(),
        "cv_brier_mean": -cv_res["test_brier"].mean(),
        "val_pr_auc": val_tuned["pr_auc"],
        "val_roc_auc": val_tuned["roc_auc"],
        "val_brier": val_tuned["brier"],
        "val_ece": val_tuned["ece"],
        "val_f1_at_0.5": val_05["f1"],
        "threshold_oof": threshold,
        "val_f1_tuned": val_tuned["f1"],
        "val_precision_tuned": val_tuned["precision"],
        "val_recall_tuned": val_tuned["recall"],
        "seconds": time.time() - start,
    }
    path = MODELS / f"compare_{name}_{imbalance}.joblib"
    joblib.dump(pipe, path)
    run_id = log_run("compare", name, {"imbalance": imbalance}, metrics, str(path.relative_to(ROOT)))
    rows.append({"model": name, "imbalance": imbalance, **metrics, "run_id": run_id})
    print(f"{name:14s} {imbalance:9s} cv PR-AUC {metrics['cv_pr_auc_mean']:.3f} "
          f"val F1@0.5 {metrics['val_f1_at_0.5']:.3f} val F1 tuned {metrics['val_f1_tuned']:.3f} "
          f"brier {metrics['val_brier']:.3f}")

results = pd.DataFrame(rows)
results.to_csv(TAB / "model_comparison.csv", index=False)

ablation = []
for name in ["logreg", "lightgbm"]:
    for engineered in [False, True]:
        pipe = build_pipeline(name, engineered=engineered)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        metrics = {"cv_pr_auc_mean": cv_res["test_pr_auc"].mean(),
                   "cv_pr_auc_std": cv_res["test_pr_auc"].std(),
                   "cv_roc_auc_mean": cv_res["test_roc_auc"].mean(),
                   "cv_brier_mean": -cv_res["test_brier"].mean()}
        log_run("ablation", name, {"engineered": engineered}, metrics, "reports/tables/feature_ablation.csv")
        ablation.append({"model": name, "features": "engineered" if engineered else "raw", **metrics})
ablation = pd.DataFrame(ablation)
ablation.to_csv(TAB / "feature_ablation.csv", index=False)
print(ablation.round(4).to_string(index=False))
