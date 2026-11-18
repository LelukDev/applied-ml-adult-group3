import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold

from common import best_threshold, get_metrics, get_splits, log_run, tuned_pipeline
from config import FIG, MODELS, N_FOLDS, ROOT, RUNS, SEED, TAB

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# all candidates are fitted on the training set and compared on the validation set
candidates = {
    "raw": tuned_pipeline(),
    "class_weights": tuned_pipeline(class_weight="balanced"),
    "platt": CalibratedClassifierCV(tuned_pipeline(), method="sigmoid", cv=cv),
    "isotonic": CalibratedClassifierCV(tuned_pipeline(), method="isotonic", cv=cv),
}

rows, probs = [], {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_val)[:, 1]
    probs[name] = p
    m = get_metrics(y_val, p)
    path = MODELS / f"calibration_{name}.joblib"
    joblib.dump(model, path)
    log_run("calibration", name, {"candidate": name}, {f"val_{k}": v for k, v in m.items()},
            str(path.relative_to(ROOT)))
    rows.append({"candidate": name, "brier": m["brier"], "ece": m["ece"], "pr_auc": m["pr_auc"],
                 "roc_auc": m["roc_auc"], "mean_pred": p.mean(), "observed_rate": y_val.mean()})
table = pd.DataFrame(rows)
table.to_csv(TAB / "calibration_val.csv", index=False)
print(table.round(4).to_string(index=False))

fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
for name, p in probs.items():
    observed, predicted = calibration_curve(y_val, p, n_bins=15, strategy="uniform")
    ax.plot(predicted, observed, "o-", ms=3, label=name)
    ax_hist.hist(p, bins=15, range=(0, 1), histtype="step", lw=1.3)
ax.set_ylabel("observed share >50K")
ax.set_title("Reliability diagram (validation set, 15 bins)")
ax.legend(fontsize=8)
ax_hist.set_yscale("log")
ax_hist.set_xlabel("predicted probability")
ax_hist.set_ylabel("n (log)")
fig.tight_layout()
fig.savefig(FIG / "calibration_reliability.png", dpi=150)

# gap between predicted and observed rate in bins of width 0.1
edges = np.linspace(0, 1, 11)
gaps = []
for name in ["raw", "class_weights"]:
    p = probs[name]
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    for b in range(10):
        mask = bins == b
        if mask.sum() >= 30:
            gaps.append({"candidate": name, "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "mean_pred": p[mask].mean(), "observed": y_val.to_numpy()[mask].mean()})
gaps = pd.DataFrame(gaps)
gaps["pred_minus_observed"] = gaps.mean_pred - gaps.observed
gaps.to_csv(TAB / "calibration_bins.csv", index=False)
print(gaps.round(3).to_string(index=False))

# final model: keep the raw model unless a candidate lowers the validation Brier score by more than 0.001
brier = table.set_index("candidate")["brier"]
best = brier.idxmin()
if brier["raw"] - brier[best] < 0.001:
    best = "raw"
threshold = best_threshold(y_val, probs[best])
joblib.dump(candidates[best], RUNS / "final_model.joblib")
(RUNS / "final_model.json").write_text(json.dumps({"candidate": best, "threshold": threshold}, indent=2))
print("final model:", best, "| threshold:", round(threshold, 3))
