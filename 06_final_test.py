import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from common import get_metrics, get_splits, log_run
from config import FIG, RUNS, SEED, TAB

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
model = joblib.load(RUNS / "final_model.joblib")
info = json.loads((RUNS / "final_model.json").read_text())
threshold = info["threshold"]

# the test set is used here for the first and only time
p_test = model.predict_proba(X_test)[:, 1]
test = get_metrics(y_test, p_test, threshold)
print("test:", {k: round(v, 4) for k, v in test.items()})

y = y_test.to_numpy()
rng = np.random.default_rng(SEED)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    m = get_metrics(y[i], p_test[i], threshold)
    boot.append({k: m[k] for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier", "ece"]})
boot = pd.DataFrame(boot)
ci = pd.DataFrame({"estimate": pd.Series({k: test[k] for k in boot.columns}),
                   "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975)})
ci.to_csv(TAB / "final_test_ci.csv")
print(ci.round(4))

prior = float(y_train.mean())
brier_ref = np.mean((prior - y) ** 2)
print("no-skill: PR-AUC", round(y.mean(), 3), "Brier", round(brier_ref, 4),
      "| Brier skill score", round(1 - test["brier"] / brier_ref, 3))

train_pr_auc = average_precision_score(y_train, model.predict_proba(X_train)[:, 1])
val_pr_auc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
print("PR-AUC train / validation / test:", round(train_pr_auc, 4), round(val_pr_auc, 4), round(test["pr_auc"], 4))

pred = (p_test >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
pd.DataFrame([{"tn": tn, "fp": fp, "fn": fn, "tp": tp}]).to_csv(TAB / "final_test_confusion.csv", index=False)

precision, recall, _ = precision_recall_curve(y, p_test)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(recall, precision, color="#0072B2", lw=2, label="final model")
axes[0].axhline(y.mean(), color="#D55E00", ls="--", label=f"no skill ({y.mean():.3f})")
axes[0].plot(test["recall"], test["precision"], "o", color="#1B2A41", ms=8, label=f"threshold {threshold:.2f}")
axes[0].set_xlabel("recall")
axes[0].set_ylabel("precision")
axes[0].set_title("Precision-recall curve (test set)")
axes[0].legend(frameon=False)
matrix = np.array([[tn, fp], [fn, tp]])
axes[1].imshow(matrix, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=14,
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
axes[1].set_xticks([0, 1], ["predicted <=50K", "predicted >50K"])
axes[1].set_yticks([0, 1], ["actual <=50K", "actual >50K"])
axes[1].set_title(f"Confusion matrix at threshold {threshold:.2f}")
fig.tight_layout()
fig.savefig(FIG / "final_test_curves.png", dpi=150)

pd.DataFrame([{"model": info["candidate"], **test, "prevalence": y.mean(), "brier_no_skill": brier_ref,
               "brier_skill_score": 1 - test["brier"] / brier_ref, "train_pr_auc": train_pr_auc,
               "val_pr_auc": val_pr_auc}]).to_csv(TAB / "final_test_metrics.csv", index=False)
log_run("final_test", info["candidate"], {"threshold": threshold},
        {f"test_{k}": v for k, v in test.items()}, "runs/final_model.joblib")
