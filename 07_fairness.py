import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import best_threshold, ece_score, get_splits, log_run, tuned_pipeline
from config import FIG, RUNS, SEED, SENSITIVE, TAB

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
model = joblib.load(RUNS / "final_model.joblib")
threshold = json.loads((RUNS / "final_model.json").read_text())["threshold"]

p_val = model.predict_proba(X_val)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]
yv, yt = y_val.to_numpy(), y_test.to_numpy()
gv = X_val[SENSITIVE].astype(str).to_numpy()
gt = X_test[SENSITIVE].astype(str).to_numpy()
rng = np.random.default_rng(SEED)


def rates(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return {"tpr": tp / max((y == 1).sum(), 1), "fpr": fp / max((y == 0).sum(), 1),
            "selection_rate": pred.mean(), "precision": tp / max(pred.sum(), 1)}


def predict(proba, groups, thresholds):
    if isinstance(thresholds, dict):
        return (proba >= np.array([thresholds[g] for g in groups])).astype(int)
    return (proba >= thresholds).astype(int)


def group_table(y, proba, groups, thresholds):
    pred = predict(proba, groups, thresholds)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        n_pos = int(y[m].sum())
        r = rates(y[m], pred[m])
        rows.append({"group": g, "n": int(m.sum()), "n_pos": n_pos, "prevalence": y[m].mean(),
                     "threshold": thresholds[g] if isinstance(thresholds, dict) else thresholds, **r,
                     "tpr_se": np.sqrt(r["tpr"] * (1 - r["tpr"]) / max(n_pos, 1)),
                     "pr_auc": average_precision_score(y[m], proba[m]) if n_pos else np.nan,
                     "roc_auc": roc_auc_score(y[m], proba[m]) if 0 < n_pos < m.sum() else np.nan,
                     "ece": ece_score(y[m], proba[m])})
    return pd.DataFrame(rows)


def gaps(y, proba, groups, thresholds):
    pred = predict(proba, groups, thresholds)
    male = rates(y[groups == "Male"], pred[groups == "Male"])
    female = rates(y[groups == "Female"], pred[groups == "Female"])
    return {"eq_opp_gap_tpr": male["tpr"] - female["tpr"], "fpr_gap": male["fpr"] - female["fpr"],
            "dp_gap": male["selection_rate"] - female["selection_rate"],
            "dp_ratio": female["selection_rate"] / male["selection_rate"]}


def bootstrap(y, proba, groups, thresholds, n_boot=1000):
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        out.append(gaps(y[i], proba[i], groups[i], thresholds))
    out = pd.DataFrame(out)
    return {k: (float(out[k].quantile(0.025)), float(out[k].quantile(0.975))) for k in out}


def overall(y, proba, groups, thresholds):
    pred = predict(proba, groups, thresholds)
    r = rates(y, pred)
    return {"accuracy": float((pred == y).mean()), "precision": r["precision"], "recall": r["tpr"],
            "f1": 2 * r["precision"] * r["tpr"] / (r["precision"] + r["tpr"])}


# 1. audit with the global threshold
before = group_table(yt, p_test, gt, threshold)
gap_before = gaps(yt, p_test, gt, threshold)
ci_before = bootstrap(yt, p_test, gt, threshold)
print(before.round(4).to_string(index=False))
print({k: round(v, 4) for k, v in gap_before.items()}, ci_before)

# 2. mitigation: one threshold per sex, chosen on validation so that each group
# reaches the overall validation recall of the original policy
target_tpr = rates(yv, (p_val >= threshold).astype(int))["tpr"]
group_thr = {}
for g in np.unique(gv):
    scores = p_val[(gv == g) & (yv == 1)]
    group_thr[g] = float(np.quantile(scores, 1 - target_tpr))
print("target recall:", round(target_tpr, 3), "group thresholds:", group_thr)

after = group_table(yt, p_test, gt, group_thr)
gap_after = gaps(yt, p_test, gt, group_thr)
ci_after = bootstrap(yt, p_test, gt, group_thr)
print(after.round(4).to_string(index=False))
print({k: round(v, 4) for k, v in gap_after.items()}, ci_after)

# 3. simple alternative: retrain without the sex column
unaware = tuned_pipeline(drop=(SENSITIVE,)).fit(X_train, y_train)
p_val_u = unaware.predict_proba(X_val)[:, 1]
p_test_u = unaware.predict_proba(X_test)[:, 1]
threshold_u = best_threshold(yv, p_val_u)
gap_unaware = gaps(yt, p_test_u, gt, threshold_u)
ci_unaware = bootstrap(yt, p_test_u, gt, threshold_u)
unaware_table = group_table(yt, p_test_u, gt, threshold_u)
print(unaware_table.round(4).to_string(index=False))

summary = pd.DataFrame([
    {"policy": "global threshold", **overall(yt, p_test, gt, threshold), **gap_before},
    {"policy": "no sex column", **overall(yt, p_test_u, gt, threshold_u), **gap_unaware},
    {"policy": "group thresholds", **overall(yt, p_test, gt, group_thr), **gap_after},
])
summary.to_csv(TAB / "fairness_summary_test.csv", index=False)
pd.concat([before.assign(policy="before"), after.assign(policy="after"),
           unaware_table.assign(policy="no sex column")]).to_csv(TAB / "fairness_sex_test.csv", index=False)
(TAB / "fairness_ci.json").write_text(json.dumps(
    {"before": ci_before, "after": ci_after, "no_sex_column": ci_unaware,
     "group_thresholds": group_thr, "target_tpr": target_tpr,
     "validation_gap_before": gaps(yv, p_val, gv, threshold)["eq_opp_gap_tpr"]}, indent=2))
print(summary.round(4).to_string(index=False))
for _, row in summary.iterrows():
    log_run("fairness", row["policy"], {"sensitive": SENSITIVE},
            {k: v for k, v in row.items() if k != "policy"}, "reports/tables/fairness_summary_test.csv")

calibration = []
for g in np.unique(gt):
    m = gt == g
    calibration.append({"group": g, "mean_pred": float(p_test[m].mean()),
                        "observed": float(yt[m].mean()), "ece": ece_score(yt[m], p_test[m])})
pd.DataFrame(calibration).to_csv(TAB / "fairness_calibration_by_sex.csv", index=False)
print(pd.DataFrame(calibration).round(4))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
policies = [(before, "global threshold"), (unaware_table, "no sex column"), (after, "group thresholds")]
for ax, metric, title in zip(axes, ["tpr", "fpr", "selection_rate"],
                             ["Recall of eligible people (TPR)", "False positive rate", "Selection rate"]):
    x = np.arange(2)
    for k, (table, label) in enumerate(policies):
        err = 1.96 * table["tpr_se"] if metric == "tpr" else None
        ax.bar(x + (k - 1) * 0.27, table[metric], width=0.27, label=label, yerr=err, capsize=3)
    ax.set_xticks(x, before["group"])
    ax.set_title(title)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.suptitle("Fairness by sex on the test set")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fairness_sex.png", dpi=150)

# paired bootstrap: how much does each fix reduce the gap, on the same resamples of the test set
reductions = []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    before_gap = gaps(yt[i], p_test[i], gt[i], threshold)["eq_opp_gap_tpr"]
    no_sex_gap = gaps(yt[i], p_test_u[i], gt[i], threshold_u)["eq_opp_gap_tpr"]
    group_gap = gaps(yt[i], p_test[i], gt[i], group_thr)["eq_opp_gap_tpr"]
    reductions.append({"no_sex_column": before_gap - no_sex_gap, "group_thresholds": before_gap - group_gap})
reductions = pd.DataFrame(reductions)
paired = {k: [float(reductions[k].quantile(0.025)), float(reductions[k].quantile(0.975))] for k in reductions}
(TAB / "fairness_paired_ci.json").write_text(json.dumps(paired, indent=2))
print("reduction of the gap, 95% interval:", paired)
