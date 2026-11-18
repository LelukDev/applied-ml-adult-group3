import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import best_threshold, build_pipeline, get_metrics, get_splits, log_run
from config import FIG, MODELS, N_FOLDS, N_TRIALS, ROOT, RUNS, SEED, TAB

optuna.logging.set_verbosity(optuna.logging.WARNING)

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train))


def suggest(name, trial):
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "solver": "saga", "max_iter": 3000, "tol": 1e-3}
    if name == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 40, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
                "max_features": trial.suggest_float("max_features", 0.1, 0.8),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0)}
    return {"n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}


def make_objective(name):
    def objective(trial):
        params = suggest(name, trial)
        start = time.time()
        pr_scores, roc_scores = [], []
        for train_idx, test_idx in folds:
            pipe = build_pipeline(name, **params)
            pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            p = pipe.predict_proba(X_train.iloc[test_idx])[:, 1]
            pr_scores.append(average_precision_score(y_train.iloc[test_idx], p))
            roc_scores.append(roc_auc_score(y_train.iloc[test_idx], p))
        metrics = {"cv_pr_auc_mean": float(np.mean(pr_scores)), "cv_pr_auc_std": float(np.std(pr_scores)),
                   "cv_roc_auc_mean": float(np.mean(roc_scores)), "fold_pr_auc": pr_scores,
                   "seconds": time.time() - start}
        log_run("optuna", name, {"trial": trial.number, **params}, metrics,
                f"reports/tables/optuna_trials_{name}.csv; reports/figures/optuna_history.png")
        return float(np.mean(pr_scores))
    return objective


studies, best_params = {}, {}
for name, n_trials in N_TRIALS.items():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(name), n_trials=n_trials)
    studies[name] = study
    fixed = {"logreg": {"solver": "saga", "max_iter": 3000, "tol": 1e-3},
             "random_forest": {}, "lightgbm": {"subsample_freq": 1}}[name]
    best_params[name] = {**study.best_params, **fixed}
    study.trials_dataframe().to_csv(TAB / f"optuna_trials_{name}.csv", index=False)
    importance = optuna.importance.get_param_importances(study)
    (TAB / f"optuna_param_importance_{name}.json").write_text(json.dumps(importance, indent=2))
    (RUNS / f"best_params_{name}.json").write_text(json.dumps(best_params[name], indent=2))
    print(name, "best CV PR-AUC", round(study.best_value, 4), "trial", study.best_trial.number)
    print(best_params[name])

# tuned models, same protocol as 02: threshold from out-of-fold predictions, metrics on validation
default = pd.read_csv(TAB / "model_comparison.csv").query("imbalance == 'none'").set_index("model")
rows = []
for name, params in best_params.items():
    pipe = build_pipeline(name, **params)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof)
    pipe.fit(X_train, y_train)
    val = get_metrics(y_val, pipe.predict_proba(X_val)[:, 1], threshold)
    path = MODELS / f"tuned_{name}.joblib"
    joblib.dump(pipe, path)
    row = {"model": name, "n_trials": len(studies[name].trials),
           "cv_pr_auc_default": default.loc[name, "cv_pr_auc_mean"],
           "cv_pr_auc_tuned": studies[name].best_value,
           "val_pr_auc_default": default.loc[name, "val_pr_auc"], "val_pr_auc_tuned": val["pr_auc"],
           "val_roc_auc_tuned": val["roc_auc"],
           "val_f1_default": default.loc[name, "val_f1_tuned"], "val_f1_tuned": val["f1"],
           "val_brier_default": default.loc[name, "val_brier"], "val_brier_tuned": val["brier"],
           "val_ece_tuned": val["ece"], "threshold_oof": threshold,
           "best_trial": studies[name].best_trial.number}
    rows.append(row)
    log_run("tuned", name, params, {k: v for k, v in row.items() if k != "model"},
            str(path.relative_to(ROOT)))
tuned = pd.DataFrame(rows)
tuned.to_csv(TAB / "tuned_comparison.csv", index=False)
print(tuned.round(4).to_string(index=False))

# the family is chosen with the training CV score only
winner = tuned.sort_values("cv_pr_auc_tuned", ascending=False).iloc[0]["model"]
(RUNS / "best_model.json").write_text(json.dumps({"model": winner, "params": best_params[winner]}, indent=2))
print("selected family:", winner)

titles = {"logreg": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM"}
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3), sharey=True)
for ax, (name, study) in zip(axes, studies.items()):
    trials = study.trials_dataframe()
    ax.scatter(trials.number, trials.value, s=12, alpha=0.7, label="trial")
    ax.plot(trials.number, trials.value.cummax(), color="C1", lw=2, label="best so far")
    ax.axhline(default.loc[name, "cv_pr_auc_mean"], color="k", ls=":", lw=1, label="default")
    ax.set_title(titles[name])
    ax.set_xlabel("trial")
axes[0].set_ylabel("CV PR-AUC (5 folds)")
axes[0].set_ylim(0.70, 0.84)
axes[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "optuna_history.png", dpi=150)
