import json
import re

import pandas as pd

from config import RUNS, ROOT, TAB

OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {"logreg": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM",
         "dummy": "No skill (prior)"}
IMBALANCE = {"none": "none", "balanced": "class weights", "smote": "SMOTE"}


def write_table(name, header, rows, align):
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def f3(x):
    return f"{x:.3f}"


def interval(pair):
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


split = pd.read_csv(TAB / "split_sizes.csv", index_col=0)
write_table("table_split", ["Split", "People", "Share >50K"],
            [[name, f"{int(r.n):,}".replace(",", "\\,"), f"{100 * r.positive_rate:.1f}\\%"]
             for name, r in split.iterrows()], "lrr")

ablation = pd.read_csv(TAB / "feature_ablation.csv")
write_table("table_ablation", ["Model", "Features", "CV PR-AUC", "sd", "CV ROC-AUC"],
            [[NAMES[r.model], r.features, f3(r.cv_pr_auc_mean), f3(r.cv_pr_auc_std), f3(r.cv_roc_auc_mean)]
             for r in ablation.itertuples()], "llrrr")

models = pd.read_csv(TAB / "model_comparison.csv").rename(columns={"val_f1_at_0.5": "val_f1_at_05"})
write_table("table_models",
            ["Model", "Imbalance", "CV PR-AUC", "Val ROC-AUC", "Val F1 at 0.5", "Val F1 tuned thr.", "Val Brier"],
            [[NAMES[r.model], IMBALANCE[r.imbalance], f3(r.cv_pr_auc_mean), f3(r.val_roc_auc),
              f3(r.val_f1_at_05), f3(r.val_f1_tuned), f3(r.val_brier)] for r in models.itertuples()], "llrrrrr")

tuned = pd.read_csv(TAB / "tuned_comparison.csv")
write_table("table_tuned",
            ["Model", "Trials", "CV PR-AUC default", "CV PR-AUC tuned", "Val PR-AUC default",
             "Val PR-AUC tuned", "Val F1 tuned", "Val Brier tuned"],
            [[NAMES[r.model], int(r.n_trials), f3(r.cv_pr_auc_default), f3(r.cv_pr_auc_tuned),
              f3(r.val_pr_auc_default), f3(r.val_pr_auc_tuned), f3(r.val_f1_tuned), f3(r.val_brier_tuned)]
             for r in tuned.itertuples()], "lrrrrrrr")

LABELS = {"raw": "LightGBM as is", "class_weights": "with class weights",
          "platt": "Platt scaling", "isotonic": "isotonic regression"}
cal = pd.read_csv(TAB / "calibration_val.csv")
write_table("table_calibration", ["Candidate", "Brier", "ECE", "PR-AUC", "Mean prediction"],
            [[LABELS[r.candidate], f"{r.brier:.4f}", f"{r.ece:.4f}", f3(r.pr_auc), f3(r.mean_pred)]
             for r in cal.itertuples()], "lrrrr")

ci = pd.read_csv(TAB / "final_test_ci.csv", index_col=0)
label = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1", "precision": "Precision",
         "recall": "Recall", "accuracy": "Accuracy", "brier": "Brier", "ece": "ECE"}
write_table("table_test", ["Metric", "Estimate", "95\\% interval"],
            [[label[k], f3(r.estimate), interval((r.ci_low, r.ci_high))] for k, r in ci.iterrows()], "lrr")

groups = pd.read_csv(TAB / "fairness_sex_test.csv")
order = {"before": 0, "no sex column": 1, "after": 2}
text = {"before": "Global threshold", "after": "Group thresholds", "no sex column": "No sex column"}
groups = groups.assign(o=groups.policy.map(order)).sort_values(["o", "group"])
write_table("table_fair_groups", ["Policy", "Group", "People (positives)", "Threshold", "TPR", "FPR",
                                  "Precision", "Selection"],
            [[text[r.policy], r.group, f"{r.n} ({r.n_pos})", f"{r.threshold:.2f}", f3(r.tpr), f3(r.fpr),
              f3(r.precision), f3(r.selection_rate)] for r in groups.itertuples()], "llrrrrrr")

summary = pd.read_csv(TAB / "fairness_summary_test.csv")
fair_ci = json.loads((TAB / "fairness_ci.json").read_text())
key = {"global threshold": "before", "no sex column": "no_sex_column", "group thresholds": "after"}
rows = []
for r in summary.itertuples():
    gap_ci = fair_ci[key[r.policy]]["eq_opp_gap_tpr"]
    rows.append([r.policy.capitalize(), f3(r.accuracy), f3(r.f1), f3(r.precision), f3(r.recall),
                 f"{r.eq_opp_gap_tpr:.3f} {interval(gap_ci)}", f3(r.fpr_gap), f3(r.dp_ratio)])
write_table("table_fair_policies",
            ["Policy", "Accuracy", "F1", "Precision", "Recall", "TPR gap [95\\% interval]", "FPR gap", "DP ratio"],
            rows, "lrrrrrrr")

log = pd.read_csv(RUNS / "run_log.csv")
what = {"compare": "no skill + 3 models x 3 imbalance strategies", "ablation": "raw vs engineered features",
        "optuna": "one row per trial", "tuned": "best trial of each model, refitted",
        "calibration": "calibration candidates", "final_test": "final model on the test set",
        "fairness": "three decision policies"}
rows = [[stage.replace("_", "\\_"), int((log.stage == stage).sum()), what[stage]]
        for stage in ["compare", "ablation", "optuna", "tuned", "calibration", "final_test", "fairness"]]
rows.append(["total", len(log), ""])
write_table("table_runlog", ["Stage", "Runs", "Content"], rows, "lrl")
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))


# ---- numbers used in the text of the report -------------------------------------------------
macros = {}


def put(name, value):
    assert name.isalpha(), name
    macros[name] = value


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def n(x):
    return f"{int(x):,}".replace(",", "\\,")


facts = pd.read_csv(TAB / "eda_facts.csv", index_col=0)["value"]
put("nRaw", n(facts.n_raw))
put("nDup", n(facts.n_duplicates))
put("nClean", n(facts.n_after_duplicates))
put("posMarried", pct(facts.pos_rate_married, 0))
put("posNotMarried", pct(facts.pos_rate_not_married, 0))
put("posMale", pct(facts.pos_rate_male))
put("posFemale", pct(facts.pos_rate_female))
put("husbandMen", pct(facts.husband_share_men, 0))
put("partTimeMen", pct(facts.part_time_share_men, 0))
put("partTimeWomen", pct(facts.part_time_share_women, 0))
put("capGainZero", pct(facts.capital_gain_zero_pct / 100))
put("capLossZero", pct(facts.capital_loss_zero_pct / 100))

for split_name, row in split.iterrows():
    put("n" + split_name.capitalize(), n(row.n))
put("posRate", pct(split.loc["train", "positive_rate"]))

miss = pd.read_csv(TAB / "eda_missing.csv").set_index("column")
put("missWork", pct(miss.loc["workclass", "pct_missing"] / 100))
put("missOcc", pct(miss.loc["occupation", "pct_missing"] / 100))
put("missCountry", pct(miss.loc["native-country", "pct_missing"] / 100))
put("posRateMissOcc", pct(miss.loc["occupation", "pos_rate_if_missing"]))
put("posRatePresentOcc", pct(miss.loc["occupation", "pos_rate_if_present"]))

short = {"logreg": "Log", "random_forest": "Rf", "lightgbm": "Lgb", "dummy": "Dummy"}
abl = ablation.set_index(["model", "features"])
for m in ["logreg", "lightgbm"]:
    put("abl" + short[m] + "Raw", f3(abl.loc[(m, "raw"), "cv_pr_auc_mean"]))
    put("abl" + short[m] + "Eng", f3(abl.loc[(m, "engineered"), "cv_pr_auc_mean"]))
    put("abl" + short[m] + "Sd", f3(abl.loc[(m, "raw"), "cv_pr_auc_std"]))
    gain = abl.loc[(m, "engineered"), "cv_pr_auc_mean"] - abl.loc[(m, "raw"), "cv_pr_auc_mean"]
    put("abl" + short[m] + "Gain", f"{100 * gain:.1f}")

imb = {"none": "None", "balanced": "Bal", "smote": "Smote"}
for r in models.itertuples():
    tag = short[r.model] + imb[r.imbalance]
    put("cv" + tag, f3(r.cv_pr_auc_mean))
    put("brier" + tag, f3(r.val_brier))
    put("fHalf" + tag, f3(r.val_f1_at_05))
    put("fTuned" + tag, f3(r.val_f1_tuned))
put("cvDummyNone", f3(models.query("model == 'dummy'").cv_pr_auc_mean.iloc[0]))
none_scores = models[models.imbalance == "none"].set_index("model").cv_pr_auc_mean
losses = [none_scores[r.model] - r.cv_pr_auc_mean for r in models.itertuples()
          if r.model != "dummy" and r.imbalance != "none"]
put("remedyLossMax", f"{100 * max(losses):.1f}")

for r in tuned.itertuples():
    s = short[r.model]
    put("tunedCv" + s, f3(r.cv_pr_auc_tuned))
    put("defaultCv" + s, f3(r.cv_pr_auc_default))
    put("tunedVal" + s, f3(r.val_pr_auc_tuned))
    put("defaultVal" + s, f3(r.val_pr_auc_default))
    put("tunedGain" + s, f"{100 * (r.cv_pr_auc_tuned - r.cv_pr_auc_default):.1f}")
    put("bestTrial" + s, int(r.best_trial))
for m in ["logreg", "random_forest", "lightgbm"]:
    params = json.loads((RUNS / f"best_params_{m}.json").read_text())
    for k, v in params.items():
        put("best" + short[m] + "".join(w.capitalize() for w in k.replace("l1", "lOne").split("_")),
            f"{v:.3g}" if isinstance(v, float) else str(v))
    trials = pd.read_csv(TAB / f"optuna_trials_{m}.csv")
    put("nTrials" + short[m], len(trials))
importance = json.loads((TAB / "optuna_param_importance_lightgbm.json").read_text())
top = sorted(importance.items(), key=lambda kv: -kv[1])[:2]
put("lgbImpOneName", top[0][0].replace("_", "\\_"))
put("lgbImpOne", f"{top[0][1]:.2f}")
put("lgbImpTwoName", top[1][0].replace("_", "\\_"))
put("lgbImpTwo", f"{top[1][1]:.2f}")

cal = cal.set_index("candidate")
for c, tag in [("raw", "Raw"), ("class_weights", "Weights"), ("platt", "Platt"), ("isotonic", "Iso")]:
    put("brierCal" + tag, f"{cal.loc[c, 'brier']:.4f}")
    put("eceCal" + tag, f"{cal.loc[c, 'ece']:.3f}")
put("meanPredRaw", pct(cal.loc["raw", "mean_pred"]))
put("meanPredWeights", pct(cal.loc["class_weights", "mean_pred"]))
put("obsRate", pct(cal.loc["raw", "observed_rate"]))

test = pd.read_csv(TAB / "final_test_metrics.csv").iloc[0]
for k, tag in [("pr_auc", "PrAuc"), ("roc_auc", "RocAuc"), ("f1", "FOne"), ("precision", "Precision"),
               ("recall", "Recall"), ("accuracy", "Acc"), ("brier", "Brier"), ("ece", "Ece")]:
    put("test" + tag, f3(test[k]))
    put("test" + tag + "Low", f3(ci.loc[k, "ci_low"]))
    put("test" + tag + "High", f3(ci.loc[k, "ci_high"]))
put("testThr", f"{test.threshold:.2f}")
put("brierNoSkill", f3(test.brier_no_skill))
put("brierSkill", f"{test.brier_skill_score:.2f}")
put("prevalenceTest", pct(test.prevalence))

g = groups.set_index(["policy", "group"])
for pol, tag in [("before", "Before"), ("after", "After"), ("no sex column", "NoSex")]:
    for sex in ["Female", "Male"]:
        row = g.loc[(pol, sex)]
        put("tpr" + sex + tag, f3(row.tpr))
        put("fpr" + sex + tag, f3(row.fpr))
        put("prec" + sex + tag, f3(row.precision))
        put("thr" + sex + tag, f"{row.threshold:.2f}")
put("nFemaleTest", n(g.loc[("before", "Female"), "n"]))
put("nMaleTest", n(g.loc[("before", "Male"), "n"]))
put("posFemaleTest", n(g.loc[("before", "Female"), "n_pos"]))
put("prevFemaleTest", pct(g.loc[("before", "Female"), "prevalence"]))
put("prevMaleTest", pct(g.loc[("before", "Male"), "prevalence"]))
put("rocFemale", f3(g.loc[("before", "Female"), "roc_auc"]))
put("rocMale", f3(g.loc[("before", "Male"), "roc_auc"]))
put("eceFemale", f3(g.loc[("before", "Female"), "ece"]))
put("eceMale", f3(g.loc[("before", "Male"), "ece"]))

sm = summary.set_index("policy")
for pol, tag in [("global threshold", "Before"), ("group thresholds", "After"), ("no sex column", "NoSex")]:
    k = key[pol]
    put("gap" + tag, f3(sm.loc[pol, "eq_opp_gap_tpr"]))
    put("gapLow" + tag, f3(fair_ci[k]["eq_opp_gap_tpr"][0]))
    put("gapHigh" + tag, f3(fair_ci[k]["eq_opp_gap_tpr"][1]))
    put("acc" + tag, f3(sm.loc[pol, "accuracy"]))
    put("fOne" + tag, f3(sm.loc[pol, "f1"]))
    put("fprGap" + tag, f3(sm.loc[pol, "fpr_gap"]))
    put("dpRatio" + tag, f"{sm.loc[pol, 'dp_ratio']:.2f}")
put("gapPointsBefore", f"{100 * sm.loc['global threshold', 'eq_opp_gap_tpr']:.1f}")
put("gapPointsAfter", f"{100 * sm.loc['group thresholds', 'eq_opp_gap_tpr']:.1f}")
female_before, female_after = g.loc[("before", "Female")], g.loc[("after", "Female")]
extra_contacted = (female_after.selection_rate - female_before.selection_rate) * female_before.n
extra_eligible = (female_after.tpr - female_before.tpr) * female_before.n_pos
put("extraWomenContacted", int(round(extra_contacted)))
put("extraWomenEligible", int(round(extra_eligible)))
put("extraWomenShare", pct(extra_eligible / extra_contacted, 0))
put("targetTpr", f3(fair_ci["target_tpr"]))

cal_sex = pd.read_csv(TAB / "fairness_calibration_by_sex.csv").set_index("group")
for sex in ["Female", "Male"]:
    put("meanPred" + sex, pct(cal_sex.loc[sex, "mean_pred"]))
    put("observed" + sex, pct(cal_sex.loc[sex, "observed"]))

shap_vars = pd.read_csv(TAB / "shap_global_grouped.csv", index_col=0)["mean_abs_shap"]
for i, (name, value) in enumerate(shap_vars.head(6).items()):
    word = ["One", "Two", "Three", "Four", "Five", "Six"][i]
    put("shap" + word + "Name", name.replace("_", "\\_"))
    put("shap" + word, f"{value:.2f}")
perm = pd.read_csv(TAB / "permutation_importance.csv", index_col=0)["mean_drop_pr_auc"]
for i, (name, value) in enumerate(perm.head(3).items()):
    word = ["One", "Two", "Three"][i]
    put("perm" + word + "Name", name)
    put("perm" + word, f"{value:.2f}")

final_info = json.loads((RUNS / "final_model.json").read_text())
final_text = {"raw": "without recalibration", "platt": "with Platt scaling",
              "isotonic": "with isotonic regression", "class_weights": "trained with class weights"}
put("finalName", final_text[final_info["candidate"]])
put("valPrAucFinal", f3(cal.loc[final_info["candidate"], "pr_auc"]))
put("valBrierFinal", f"{cal.loc[final_info['candidate'], 'brier']:.4f}")

cases = json.loads((TAB / "shap_local_cases.json").read_text())
case_tag = {"true_positive": "Tp", "false_negative": "Fn", "false_positive": "Fp", "borderline": "Bl"}
for case, tag in case_tag.items():
    info = cases[case]
    put("local" + tag + "P", f"{info['proba']:.2f}")
    put("person" + tag + "Age", f"{info['raw_row']['age']:.0f}")
    put("person" + tag + "Hours", f"{info['raw_row']['hours-per-week']:.0f}")
    put("person" + tag + "Gain", n(info["raw_row"]["capital-gain"]))
    for var, value in info["top_contributions"].items():
        camel = "".join(w.capitalize() for w in re.split("[_-]", var))
        put("local" + tag + camel, f"+{value:.2f}" if value >= 0 else f"$-${abs(value):.2f}")

paired = json.loads((TAB / "fairness_paired_ci.json").read_text())
for k, tag in [("no_sex_column", "NoSex"), ("group_thresholds", "Group")]:
    put("paired" + tag, f"{100 * (sm.loc['global threshold', 'eq_opp_gap_tpr'] - sm.loc[{'no_sex_column': 'no sex column', 'group_thresholds': 'group thresholds'}[k], 'eq_opp_gap_tpr']):.1f}")
    put("paired" + tag + "Low", f"{100 * paired[k][0]:.1f}")
    put("paired" + tag + "High", f"{100 * paired[k][1]:.1f}")
put("gapValBefore", f3(fair_ci["validation_gap_before"]))

val_change = (tuned.val_pr_auc_tuned - tuned.val_pr_auc_default).abs().max()
put("maxValGain", f"{100 * val_change:.1f}")
put("trainPrAuc", f3(test.train_pr_auc))
conf = pd.read_csv(TAB / "final_test_confusion.csv").iloc[0]
put("confTp", n(conf.tp))
put("confFp", n(conf.fp))
put("confFn", n(conf.fn))
put("confTn", n(conf.tn))
put("confPositives", n(conf.tp + conf.fn))
put("confFlagged", n(conf.tp + conf.fp))

trial_rows = log[log.stage == "optuna"].copy()
trial_rows["trial"] = trial_rows.params.apply(lambda p: json.loads(p)["trial"])
trial_rows = trial_rows[["run_id", "model", "trial", "params", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
                         "fold_pr_auc", "artifact"]]
trial_rows.to_csv(RUNS / "tuning_log.csv", index=False)

put("nRuns", len(log))
put("nOptunaRuns", int((log.stage == "optuna").sum()))

lines = ["% generated by make_report_tables.py, do not edit by hand"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
print(len(macros), "macros written")
