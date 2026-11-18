import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from common import CAT_ENG, add_features, best_threshold, get_splits, tuned_pipeline
from config import FIG, SEED, TAB

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
pipe = tuned_pipeline().fit(X_train, y_train)
model = pipe.named_steps["model"]
prep_pipe = pipe[:-1]
names = list(pipe.named_steps["prep"].get_feature_names_out())

p_val = pipe.predict_proba(X_val)[:, 1]
threshold = best_threshold(y_val, p_val)

X_enc = pd.DataFrame(prep_pipe.transform(X_val), columns=names, index=X_val.index)
shap_values = shap.TreeExplainer(model)(X_enc)
if shap_values.values.ndim == 3:
    shap_values = shap_values[:, :, 1]
base = float(np.ravel(shap_values.base_values)[0])
print("base value (log-odds):", round(base, 3), "-> probability", round(1 / (1 + np.exp(-base)), 3),
      "| mean prediction:", round(p_val.mean(), 3))

# sum the one-hot columns back into the original variable
group_of = [next((c for c in CAT_ENG if n.startswith(c + "_")), n) for n in names]
variables = list(dict.fromkeys(group_of))
M = np.zeros((len(names), len(variables)))
for j, g in enumerate(group_of):
    M[j, variables.index(g)] = 1
grouped = shap_values.values @ M

shown = add_features(X_val)[variables]
explanation = shap.Explanation(values=grouped, base_values=shap_values.base_values,
                               data=shown.to_numpy(dtype=object), feature_names=variables)

# direction of the effect: correlation between the value of a numeric variable and its SHAP value
direction = {}
for j, var in enumerate(variables):
    values = pd.to_numeric(shown[var], errors="coerce")
    if values.notna().all():
        direction[var] = pd.Series(values.to_numpy()).corr(pd.Series(grouped[:, j]), method="spearman")
pd.Series(direction).rename("spearman_value_vs_shap").to_csv(TAB / "shap_direction.csv")
occupation_shap = pd.Series(grouped[:, variables.index("occupation")]).groupby(shown["occupation"].to_numpy()).mean()
occupation_shap.sort_values(ascending=False).rename("mean_shap").to_csv(TAB / "shap_by_occupation.csv")

mean_abs = pd.Series(np.abs(shap_values.values).mean(0), index=names).sort_values(ascending=False)
mean_abs.rename("mean_abs_shap").to_csv(TAB / "shap_global.csv")
mean_abs_var = pd.Series(np.abs(grouped).mean(0), index=variables).sort_values(ascending=False)
mean_abs_var.rename("mean_abs_shap").to_csv(TAB / "shap_global_grouped.csv")
print(mean_abs_var.round(3).to_string())

top = mean_abs_var.head(12)[::-1]
plt.figure(figsize=(6, 4.2))
plt.barh(top.index, top.values, color="#4C72B0")
plt.xlabel("mean |SHAP value| (log-odds)")
plt.title("Global importance by variable (validation set)")
plt.tight_layout()
plt.savefig(FIG / "shap_bar.png", dpi=150)
plt.close("all")

plt.figure()
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("SHAP beeswarm (validation set, encoded features)")
plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close("all")

perm = permutation_importance(pipe, X_val, y_val, scoring="average_precision",
                              n_repeats=10, random_state=SEED, n_jobs=1)
perm = pd.DataFrame({"mean_drop_pr_auc": perm.importances_mean, "std": perm.importances_std},
                    index=X_val.columns).sort_values("mean_drop_pr_auc", ascending=False)
perm.to_csv(TAB / "permutation_importance.csv")
print(perm.round(4).to_string())

# four individual predictions: a typical false negative, a typical false positive,
# the person closest to the threshold and the most confident true positive
pred = (p_val >= threshold).astype(int)
y = y_val.to_numpy()


def median_case(idx):
    return int(idx[np.argsort(p_val[idx])[len(idx) // 2]])


tp = np.where((y == 1) & (pred == 1))[0]
fn = np.where((y == 1) & (pred == 0))[0]
fp = np.where((y == 0) & (pred == 1))[0]
cases = {"true_positive": int(tp[np.argmax(p_val[tp])]),
         "false_negative": median_case(fn),
         "false_positive": median_case(fp),
         "borderline": int(np.argmin(np.abs(p_val - threshold)))}

local = {}
for label, i in cases.items():
    plt.figure()
    shap.plots.waterfall(explanation[i], max_display=9, show=False)
    plt.title(f"{label.replace('_', ' ')} (y={y[i]}): p = {p_val[i]:.2f}, threshold {threshold:.2f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    top6 = pd.Series(grouped[i], index=variables).sort_values(key=np.abs, ascending=False).head(6)
    local[label] = {"proba": float(p_val[i]), "y_true": int(y[i]), "threshold": threshold,
                    "raw_row": {k: (v if isinstance(v, str) else float(v))
                                for k, v in X_val.iloc[i].items() if pd.notna(v)},
                    "top_contributions": top6.round(3).to_dict()}
(TAB / "shap_local_cases.json").write_text(json.dumps(local, indent=2, default=str))
print(json.dumps(local, indent=2, default=str))
