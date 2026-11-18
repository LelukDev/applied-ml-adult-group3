import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CAT_COLS, NUM_COLS, get_splits, load_data
from config import FIG, TAB

raw = load_data()
n_dup = int(raw.duplicated().sum())
print("rows:", len(raw), "| duplicated rows:", n_dup)

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
split = pd.DataFrame({"n": [len(y_train), len(y_val), len(y_test)],
                      "positive_rate": [y_train.mean(), y_val.mean(), y_test.mean()]},
                     index=["train", "val", "test"])
split.to_csv(TAB / "split_sizes.csv")
print(split.round(4))

# everything below uses the training set only
df = X_train.copy()
df["y"] = y_train

missing = []
for col in CAT_COLS:
    is_missing = df[col].isna()
    if is_missing.any():
        missing.append({"column": col, "n_missing": int(is_missing.sum()),
                        "pct_missing": 100 * is_missing.mean(),
                        "pos_rate_if_missing": df.loc[is_missing, "y"].mean(),
                        "pos_rate_if_present": df.loc[~is_missing, "y"].mean()})
missing = pd.DataFrame(missing)
missing.to_csv(TAB / "eda_missing.csv", index=False)
print(missing.round(3).to_string(index=False))

groups = []
for col in ["sex", "race", "marital-status", "relationship", "workclass"]:
    g = df.groupby(col)["y"].agg(["size", "mean"]).reset_index()
    g.columns = ["level", "n", "pos_rate"]
    g.insert(0, "feature", col)
    groups.append(g)
groups = pd.concat(groups)
groups.to_csv(TAB / "eda_positive_rate_by_group.csv", index=False)
print(groups[groups.feature.isin(["sex", "race"])].round(3).to_string(index=False))

summary = df[NUM_COLS].describe(percentiles=[0.5, 0.9, 0.99]).T
summary["pct_zero"] = (df[NUM_COLS] == 0).mean() * 100
summary.to_csv(TAB / "eda_numeric_summary.csv")
print(summary.round(1).to_string())

married = df["marital-status"].isin(["Married-civ-spouse", "Married-AF-spouse"])
is_husband = df["relationship"] == "Husband"
part_time = df["hours-per-week"] < 35
male, female = df.sex == "Male", df.sex == "Female"
facts = {
    "n_raw": len(raw),
    "n_duplicates": n_dup,
    "n_after_duplicates": len(raw) - n_dup,
    "pos_rate_married": df.loc[married, "y"].mean(),
    "pos_rate_not_married": df.loc[~married, "y"].mean(),
    "pos_rate_male": df.loc[male, "y"].mean(),
    "pos_rate_female": df.loc[female, "y"].mean(),
    "husband_share_men": is_husband[male].mean(),
    "husband_share_women": is_husband[female].mean(),
    "part_time_share_men": part_time[male].mean(),
    "part_time_share_women": part_time[female].mean(),
    "capital_gain_zero_pct": (df["capital-gain"] == 0).mean() * 100,
    "capital_loss_zero_pct": (df["capital-loss"] == 0).mean() * 100,
    "capital_gain_max": df["capital-gain"].max(),
}
pd.Series(facts).rename("value").to_csv(TAB / "eda_facts.csv")
print(pd.Series(facts).round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col in zip(axes, ["age", "hours-per-week", "education-num"]):
    for label, name in [(0, "<=50K"), (1, ">50K")]:
        ax.hist(df.loc[df.y == label, col], bins=30, alpha=0.6, density=True, label=name)
    ax.set_title(col)
axes[0].legend()
fig.tight_layout()
fig.savefig(FIG / "eda_numeric_by_class.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, col in zip(axes, ["sex", "relationship"]):
    rate = df.groupby(col)["y"].mean().sort_values()
    ax.barh(rate.index, rate.values, color="#4C72B0")
    ax.axvline(df.y.mean(), color="k", ls="--", lw=1, label="overall")
    ax.set_xlabel("share earning >50K")
    ax.set_title(f"Positive rate by {col}")
axes[0].legend()
fig.tight_layout()
fig.savefig(FIG / "eda_positive_rate_sex_relationship.png", dpi=150)

corr = df[NUM_COLS + ["y"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Spearman correlation")
fig.tight_layout()
fig.savefig(FIG / "eda_correlation.png", dpi=150)
