import json
import re
import sys

import numpy as np
import pandas as pd

from config import N_TRIALS, ROOT, RUNS, TAB

problems = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        problems.append(message)


split = pd.read_csv(TAB / "split_sizes.csv", index_col=0)
facts = pd.read_csv(TAB / "eda_facts.csv", index_col=0)["value"]
log = pd.read_csv(RUNS / "run_log.csv")
test = pd.read_csv(TAB / "final_test_metrics.csv").iloc[0]
ci = pd.read_csv(TAB / "final_test_ci.csv", index_col=0)
groups = pd.read_csv(TAB / "fairness_sex_test.csv")
summary = pd.read_csv(TAB / "fairness_summary_test.csv").set_index("policy")
fair_ci = json.loads((TAB / "fairness_ci.json").read_text())
cal = pd.read_csv(TAB / "calibration_val.csv").set_index("candidate")
tuned = pd.read_csv(TAB / "tuned_comparison.csv").set_index("model")
models = pd.read_csv(TAB / "model_comparison.csv")
final_info = json.loads((RUNS / "final_model.json").read_text())
best_info = json.loads((RUNS / "best_model.json").read_text())

# split
check(split.n.sum() == facts.n_after_duplicates, "split sizes do not add up to the clean data")
check(abs(split.loc["train", "n"] / split.n.sum() - 0.6) < 0.001, "train is not 60%")
check(abs(split.loc["val", "n"] / split.n.sum() - 0.2) < 0.001, "validation is not 20%")
check(split.positive_rate.max() - split.positive_rate.min() < 0.001, "positive rate differs between splits")
check(facts.n_raw - facts.n_duplicates == facts.n_after_duplicates, "duplicate count is inconsistent")

# run log
expected = {"compare": 10, "ablation": 4, "optuna": sum(N_TRIALS.values()), "tuned": 3,
            "calibration": 4, "final_test": 1, "fairness": 3}
for stage, count in expected.items():
    check((log.stage == stage).sum() == count, f"run log has {(log.stage == stage).sum()} rows for {stage}, expected {count}")
check(log.run_id.is_unique, "run ids are not unique")
for m, k in N_TRIALS.items():
    trials = pd.read_csv(TAB / f"optuna_trials_{m}.csv")
    check(len(trials) == k, f"{m}: {len(trials)} trials in the table, expected {k}")
    check(abs(trials.value.max() - tuned.loc[m, "cv_pr_auc_tuned"]) < 1e-9, f"{m}: best trial score differs from the tuned table")
    logged = log[(log.stage == "optuna") & (log.model == m)]
    check(len(logged) == k, f"{m}: run log has {len(logged)} trials")

# models
dummy = models[models.model == "dummy"].iloc[0]
check(abs(dummy.cv_pr_auc_mean - split.loc["train", "positive_rate"]) < 0.005, "no-skill PR-AUC is not close to the prevalence")
check(best_info["model"] == tuned.cv_pr_auc_tuned.idxmax(), "selected family is not the one with the best tuned CV score")

# calibration and final model
expected_final = cal.brier.idxmin() if cal.loc["raw", "brier"] - cal.brier.min() >= 0.001 else "raw"
check(final_info["candidate"] == expected_final, "final model does not follow the selection rule")
check(test["model"] == final_info["candidate"], "test model differs from the final model file")
check(abs(test.threshold - final_info["threshold"]) < 1e-9, "test threshold differs from the saved threshold")
for metric in ci.index:
    row = ci.loc[metric]
    check(row.ci_low <= row.estimate <= row.ci_high, f"{metric}: estimate outside its interval")
    check(abs(row.estimate - test[metric]) < 1e-9, f"{metric}: interval table and metrics table differ")
check(abs(test.brier_skill_score - (1 - test.brier / test.brier_no_skill)) < 1e-9, "Brier skill score is wrong")

# fairness
before = groups[groups.policy == "before"].set_index("group")
after = groups[groups.policy == "after"].set_index("group")
n_test = split.loc["test", "n"]
check(before.n.sum() == n_test, "group sizes do not add up to the test set")
check(abs(before.n_pos.sum() - test.prevalence * n_test) < 1, "group positives do not match the test prevalence")
check(before.threshold.nunique() == 1, "global threshold differs between groups")
check(abs(before.threshold.iloc[0] - test.threshold) < 1e-9, "fairness threshold differs from the final threshold")
for table in (before, after):
    for sex, r in table.iterrows():
        check(abs(r.tpr * r.n_pos - round(r.tpr * r.n_pos)) < 1e-6, f"{sex}: TPR is not a count over positives")
        check(abs(r.selection_rate * r.n - round(r.selection_rate * r.n)) < 1e-6, f"{sex}: selection rate is not a count")
sb = summary.loc["global threshold"]
sa = summary.loc["group thresholds"]
check(abs(sb.eq_opp_gap_tpr - (before.loc["Male", "tpr"] - before.loc["Female", "tpr"])) < 1e-9, "gap before is not Male TPR minus Female TPR")
check(abs(sa.eq_opp_gap_tpr - (after.loc["Male", "tpr"] - after.loc["Female", "tpr"])) < 1e-9, "gap after is not Male TPR minus Female TPR")
check(abs(sb.dp_ratio - before.loc["Female", "selection_rate"] / before.loc["Male", "selection_rate"]) < 1e-9, "DP ratio before is wrong")
check(abs(sb.recall - test.recall) < 1e-9 and abs(sb.accuracy - test.accuracy) < 1e-9 and abs(sb.f1 - test.f1) < 1e-9,
      "fairness baseline does not match the final test metrics")
for name, key in [("global threshold", "before"), ("group thresholds", "after"), ("no sex column", "no_sex_column")]:
    low, high = fair_ci[key]["eq_opp_gap_tpr"]
    check(low <= summary.loc[name, "eq_opp_gap_tpr"] <= high, f"gap interval does not contain the estimate ({name})")

# claims made in the text of the report
cv = models.set_index(["model", "imbalance"]).cv_pr_auc_mean
brier_val = models.set_index(["model", "imbalance"]).val_brier
f1_half = models.set_index(["model", "imbalance"])["val_f1_at_0.5"]
f1_tuned = models.set_index(["model", "imbalance"]).val_f1_tuned
check(cv["lightgbm", "none"] > cv["random_forest", "none"] > cv["logreg", "none"] > cv["dummy", "none"], "claim: model ranking is not LightGBM > forest > logistic > no skill")
for m in ["logreg", "random_forest", "lightgbm"]:
    for remedy in ["balanced", "smote"]:
        check(cv[m, remedy] <= cv[m, "none"] + 0.001, f"claim: {remedy} improves PR-AUC for {m}")
        check(brier_val[m, remedy] > brier_val[m, "none"], f"claim: {remedy} does not damage the Brier score for {m}")
check(f1_tuned["lightgbm", "none"] >= f1_half["lightgbm", "none"], "claim: tuning the threshold does not raise F1")
check(abs(f1_tuned["lightgbm", "none"] - f1_half["lightgbm", "smote"]) <= 0.005, "claim: tuned F1 is not as good as SMOTE at 0.5")
check(f1_tuned["lightgbm", "none"] > f1_half["lightgbm", "balanced"], "claim: tuned F1 is not better than class weights at 0.5")
abl = pd.read_csv(TAB / "feature_ablation.csv").set_index(["model", "features"])
check(abl.loc[("logreg", "engineered"), "cv_pr_auc_mean"] - abl.loc[("logreg", "raw"), "cv_pr_auc_mean"] > abl.loc[("logreg", "raw"), "cv_pr_auc_std"], "claim: engineered features do not help the logistic regression")
check(abs(abl.loc[("lightgbm", "engineered"), "cv_pr_auc_mean"] - abl.loc[("lightgbm", "raw"), "cv_pr_auc_mean"]) <= abl.loc[("lightgbm", "raw"), "cv_pr_auc_std"], "claim: engineered features change LightGBM by more than one sd")
gains = tuned.cv_pr_auc_tuned - tuned.cv_pr_auc_default
check((gains >= 0).all(), "claim: tuning lowers a CV score")
check(gains.max() < cv["lightgbm", "none"] - cv["logreg", "none"], "claim: tuning gain is not smaller than the gap between families")
check((tuned.val_pr_auc_tuned - tuned.val_pr_auc_default).abs().max() <= abl.loc[("lightgbm", "raw"), "cv_pr_auc_std"], "claim: tuning changes the validation PR-AUC by more than one fold sd")
check(json.loads((RUNS / "best_params_logreg.json").read_text())["C"] >= 1, "claim: best C is not weak regularisation")
raw_bins = pd.read_csv(TAB / "calibration_bins.csv")
for cand, sign in [("raw", None), ("class_weights", "over")]:
    b = raw_bins[raw_bins.candidate == cand]
    if sign == "over":
        check((b.pred_minus_observed > 0).all(), "claim: class weights are not over-confident in every bin")
    else:
        check((b.pred_minus_observed > 0).any() and (b.pred_minus_observed < 0).any(), "claim: raw model gaps do not have both signs")
        check(b.pred_minus_observed.abs().max() < 0.03, "claim: raw model has a bin gap above 3 points")
check(cal.loc["raw", "ece"] < 0.02, "claim: raw ECE is not small")
check(cal.loc["class_weights", "ece"] > 3 * cal.loc["raw", "ece"], "claim: class weights do not raise the ECE a lot")
check(cal.loc["raw", "brier"] - cal.loc["platt", "brier"] <= 0.001 and cal.loc["raw", "brier"] - cal.loc["isotonic", "brier"] <= 0.001, "claim: a calibrator improves the Brier score by more than 0.001")
check(final_info["candidate"] == "raw", "claim: the text says the raw model is final")
check(ci.loc["pr_auc", "ci_low"] <= cal.loc[final_info["candidate"], "pr_auc"] <= ci.loc["pr_auc", "ci_high"], "claim: validation PR-AUC is outside the test interval")
bci = fair_ci["before"]["eq_opp_gap_tpr"]
aci = fair_ci["after"]["eq_opp_gap_tpr"]
check(bci[0] > 0, "claim: gap before has an interval that contains zero")
check(aci[0] <= 0 <= aci[1], "claim: gap after has an interval that excludes zero")
check(sa.eq_opp_gap_tpr < sb.eq_opp_gap_tpr, "claim: the gap does not fall")
check(summary.loc["no sex column", "eq_opp_gap_tpr"] > sa.eq_opp_gap_tpr, "claim: removing sex is not worse than group thresholds")
check(abs(sb.eq_opp_gap_tpr - summary.loc["no sex column", "eq_opp_gap_tpr"]) < (bci[1] - bci[0]) / 2, "claim: removing sex changes the gap by more than half the interval")
check(before.loc["Female", "roc_auc"] >= before.loc["Male", "roc_auc"], "claim: women are not ranked at least as well as men")
check((before.ece < 0.03).all(), "claim: calibration by sex is not good")
check(sa.accuracy < sb.accuracy and sb.accuracy - sa.accuracy < 0.02, "claim: the accuracy cost is not small")
check(after.loc["Female", "precision"] < before.loc["Female", "precision"], "claim: precision for women does not fall")
extra_contacted = (after.loc["Female", "selection_rate"] - before.loc["Female", "selection_rate"]) * before.loc["Female", "n"]
extra_eligible = (after.loc["Female", "tpr"] - before.loc["Female", "tpr"]) * before.loc["Female", "n_pos"]
check(extra_eligible / extra_contacted > before.loc["Female", "prevalence"], "claim: extra women contacted are not above the base rate")
check(extra_eligible / extra_contacted < after.loc["Female", "precision"], "claim: extra women contacted are not less precise than the average contact")
check(sa.fpr_gap < sb.fpr_gap and sa.dp_ratio > sb.dp_ratio, "claim: FPR gap and DP ratio do not both improve")
check(sa.fpr_gap > 0.05, "claim: the FPR gap does not stay large")
shap_var = pd.read_csv(TAB / "shap_global_grouped.csv", index_col=0)["mean_abs_shap"]
top8 = list(shap_var.index[:8])
for var in ["age", "education-num", "occupation", "hours-per-week"]:
    check(var in top8, f"claim: {var} is not in the top 8 of SHAP")
check(any(v in top8 for v in ["is_married", "marital-status"]), "claim: marriage is not in the top 8 of SHAP")
check(any(v in top8 for v in ["capital-gain", "log_capital_net"]), "claim: capital gain is not in the top 8 of SHAP")
check(shap_var["sex"] < shap_var["is_married"] and shap_var["sex"] < shap_var["relationship"], "claim: sex does not have a smaller effect than marriage and relationship")
direction = pd.read_csv(TAB / "shap_direction.csv", index_col=0)["spearman_value_vs_shap"]
for var in ["age", "education-num", "hours-per-week", "is_married", "capital-gain"]:
    check(direction[var] > 0, f"claim: {var} does not push towards more than 50K")
occupation = pd.read_csv(TAB / "shap_by_occupation.csv", index_col=0)["mean_shap"]
check(occupation.index[0] == "Exec-managerial" and occupation["Prof-specialty"] > 0.3, "claim: managerial and professional jobs do not push strongly towards more than 50K")
perm = pd.read_csv(TAB / "permutation_importance.csv", index_col=0)["mean_drop_pr_auc"]
check(perm.index[0] == "capital-gain", "claim: capital gain is not first in permutation importance")
check(perm["relationship"] < perm["marital-status"], "claim: relationship is not under-stated by permutation")

conf = pd.read_csv(TAB / "final_test_confusion.csv").iloc[0]
check(conf.tp + conf.fp + conf.fn + conf.tn == n_test, "confusion matrix does not add up to the test set")
check(abs(conf.tp / (conf.tp + conf.fp) - test.precision) < 1e-9 and abs(conf.tp / (conf.tp + conf.fn) - test.recall) < 1e-9, "confusion matrix does not match precision and recall")
check(test.train_pr_auc > test.val_pr_auc, "claim: the model does not fit the training data better than new data")
check(abs(test.val_pr_auc - test.pr_auc) < 0.02, "claim: validation and test PR-AUC are not close")
paired = json.loads((TAB / "fairness_paired_ci.json").read_text())
check(paired["group_thresholds"][0] > 0, "claim: the paired gap reduction by group thresholds is not clearly positive")
check(paired["no_sex_column"][0] > 0, "claim: the paired gap reduction by dropping sex is not positive")
check(paired["no_sex_column"][1] < paired["group_thresholds"][1], "claim: dropping sex does not help less than group thresholds")
check(after.loc["Female", "tpr"] < json.loads((TAB / "fairness_ci.json").read_text())["target_tpr"], "claim: women's test recall does not fall below the target")

# report files
tex_path = ROOT / "report" / "report_group3.tex"
if tex_path.exists():
    tex = tex_path.read_text()
    numbers = (ROOT / "report" / "generated" / "numbers.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers))
    body = "\n".join(line.split("%")[0] for line in tex.splitlines())
    used = set(re.findall(r"\\([A-Za-z]+)(?![A-Za-z])", body)) & defined
    check(len(used) > 20, "the report uses very few generated numbers")
    for path in [p for p in re.findall(r"\\input\{([^}]+)\}", body) if "#" not in p]:
        check((ROOT / "report" / path).exists() or (ROOT / "report" / (path + ".tex")).exists(), f"missing input file {path}")
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        check((ROOT / "reports" / "figures" / fig).exists() or (ROOT / "report" / fig).exists(), f"missing figure {fig}")
    for bad in ["\u2014", "\u2013", "REPLACE", "TODO", "[e.g", "XXX", "LOCALTEXT", "CONTRIBUTIONTEXT", "AITEXT", "to be written"]:
        check(bad not in tex, f"the report contains '{bad}'")
    prose = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
    loose = re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", prose)
    print("decimal numbers typed by hand in the text (review them):", sorted(set(loose)))

print(f"{checks} checks run, {len(problems)} problems")
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
