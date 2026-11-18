# builds slides/presentation_group3.pptx (needs python-pptx); numbers come from report/generated/numbers.tex
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from config import FIG, ROOT, TAB

OUT = ROOT / "slides"
OUT.mkdir(exist_ok=True)

N = {}
for name, value in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (ROOT / "report/generated/numbers.tex").read_text()):
    N[name] = value.replace("\\%", "%").replace("\\,", ",").replace("\\_", "_").replace("$-$", "-")

# palette: navy text, blue main, orange accent, green for "after", greys for the rest
NAVY, BLUE, ORANGE, GREEN = "1B2A41", "0072B2", "D55E00", "009E73"
GREY, LIGHT, TEXT = "5B6575", "F2F4F7", "1F2937"
FONT = "Arial"


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": "#1F2937",
                     "xtick.color": "#1F2937", "ytick.color": "#1F2937", "axes.titlesize": 14,
                     "axes.titleweight": "bold", "axes.titlecolor": "#1B2A41"})
SIZE = (5.8, 4.7)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# ---- figures made for the slides (all numbers are read from the result tables) -----------------
groups = pd.read_csv(TAB / "eda_positive_rate_by_group.csv")
fig, axes = plt.subplots(2, 1, figsize=SIZE, gridspec_kw={"height_ratios": [2, 6]})
overall = float(N["posRate"].rstrip("%")) / 100
for ax, feature in zip(axes, ["sex", "relationship"]):
    g = groups[groups.feature == feature].sort_values("pos_rate")
    colors = ["#" + (ORANGE if level == "Female" else BLUE) for level in g.level]
    ax.barh(g.level, g.pos_rate, color=colors)
    for y, v in enumerate(g.pos_rate):
        ax.text(v + 0.008, y, f"{100 * v:.0f}%", va="center", fontsize=12)
    ax.axvline(overall, color="#5B6575", ls="--", lw=1)
    ax.set_xlim(0, 0.58)
    ax.set_xticks([])
    ax.set_title(f"Share above 50K, by {feature}", loc="left")
save(fig, "slide_eda.png")

models = pd.read_csv(TAB / "model_comparison.csv")
light = models[models.model == "lightgbm"].set_index("imbalance")
order = ["none", "balanced", "smote"]
names = ["As is", "Class weights", "SMOTE"]
cols = ["#" + BLUE, "#" + ORANGE, "#9CA3AF"]
fig, axes = plt.subplots(2, 1, figsize=SIZE)
for ax, column, title, lo, hi in [(axes[0], "cv_pr_auc_mean", "CV PR-AUC (higher is better)", 0.0, 1.0),
                                  (axes[1], "val_brier", "Validation Brier score (lower is better)", 0.0, 0.13)]:
    values = [light.loc[k, column] for k in order]
    ax.bar(names, values, color=cols, width=0.6)
    ax.set_ylim(lo, hi)
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    for i, v in enumerate(values):
        ax.text(i, v + (hi - lo) * 0.03, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
save(fig, "slide_remedies.png")

tuned = pd.read_csv(TAB / "tuned_comparison.csv")
fig, ax = plt.subplots(figsize=SIZE)
labels = ["Logistic\nregression", "Random\nforest", "LightGBM"]
x = range(3)
ax.bar([i - 0.2 for i in x], tuned.cv_pr_auc_default, width=0.38, color="#9CA3AF", label="default")
ax.bar([i + 0.2 for i in x], tuned.cv_pr_auc_tuned, width=0.38, color="#" + BLUE, label="tuned")
for i, (a, b) in enumerate(zip(tuned.cv_pr_auc_default, tuned.cv_pr_auc_tuned)):
    ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=11)
    ax.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(float(N["cvDummyNone"]), color="#" + ORANGE, ls="--", lw=1.5, label=f"no skill ({N['cvDummyNone']})")
ax.set_xticks(list(x), labels)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_title("CV PR-AUC", loc="left")
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=11)
save(fig, "slide_models.png")

trials = pd.read_csv(TAB / "optuna_trials_lightgbm.csv")
fig, ax = plt.subplots(figsize=SIZE)
ax.scatter(trials.number, trials.value, s=22, color="#9CA3AF", label="one trial")
ax.plot(trials.number, trials.value.cummax(), color="#" + BLUE, lw=2.5, label="best so far")
ax.axhline(float(N["defaultCvLgb"]), color="#" + ORANGE, ls="--", lw=1.5)
ax.text(trials.number.max(), float(N["defaultCvLgb"]) - 0.0035, "default", ha="right", color="#" + ORANGE, fontsize=12)
ax.set_ylim(0.80, 0.835)
ax.set_xlabel("Optuna trial")
ax.set_ylabel("CV PR-AUC")
ax.set_title("LightGBM: 60 trials", loc="left")
ax.legend(frameon=False, loc="lower right")
save(fig, "slide_optuna.png")

shap = pd.read_csv(TAB / "shap_global_grouped.csv", index_col=0)["mean_abs_shap"].head(8)[::-1]
fig, ax = plt.subplots(figsize=SIZE)
ax.barh(shap.index, shap.values, color="#" + BLUE)
for y, v in enumerate(shap.values):
    ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=12)
ax.set_xticks([])
ax.set_xlim(0, shap.max() * 1.15)
ax.set_title("Mean absolute SHAP value (log-odds)", loc="left")
save(fig, "slide_shap.png")

bins = pd.read_csv(TAB / "calibration_bins.csv")
fig, ax = plt.subplots(figsize=SIZE)
ax.plot([0, 1], [0, 1], color="#9CA3AF", ls="--", lw=1.5, label="perfect")
for cand, color, label in [("raw", BLUE, "model as is"), ("class_weights", ORANGE, "with class weights")]:
    b = bins[bins.candidate == cand]
    ax.plot(b.mean_pred, b.observed, "o-", color="#" + color, lw=2.5, ms=6, label=label)
ax.set_xlabel("predicted probability")
ax.set_ylabel("observed share above 50K")
ax.set_title("Reliability diagram (validation set)", loc="left")
ax.legend(frameon=False, loc="upper left")
save(fig, "slide_calibration.png")

fair = pd.read_csv(TAB / "fairness_sex_test.csv")
fig, ax = plt.subplots(figsize=SIZE)
width = 0.36
for k, (policy, label, color) in enumerate([("before", "one threshold", "#9CA3AF"),
                                            ("after", "one threshold per sex", "#" + GREEN)]):
    t = fair[fair.policy == policy].set_index("group").loc[["Female", "Male"]]
    xs = [i + (k - 0.5) * width for i in range(2)]
    ax.bar(xs, t.tpr, width=width, color=color, label=label, yerr=1.96 * t.tpr_se, capsize=4)
    for xv, v in zip(xs, t.tpr):
        ax.text(xv, 0.05, f"{v:.2f}", ha="center", color="white", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1], ["Women", "Men"])
ax.set_ylim(0, 1.15)
ax.set_yticks([])
ax.set_title("Recall of people who earn more than 50K", loc="left")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.97))
save(fig, "slide_fairness.png")

# ---- deck helpers -----------------------------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]
page = {"n": 0}


def box(slide, left, top, width, height, lines, size, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
        bullet=False, space=12):
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(lines):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT
        run.font.color.rgb = rgb(color)
        p.alignment = align
        p.space_after = Pt(space)
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.3)))
            pPr.set("indent", str(-Inches(0.3)))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "\u2022"}))
    return shape


def card(slide, left, top, width, height):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(LIGHT)
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def picture(slide, name, left, top, width, height):
    from PIL import Image
    path = FIG / name
    w, h = Image.open(path).size
    scale = min(width / w, height / h)
    pw, ph = w * scale, h * scale
    slide.shapes.add_picture(str(path), Inches(left + (width - pw) / 2), Inches(top + (height - ph) / 2),
                             Inches(pw), Inches(ph))


def new_slide(title, notes, backup=False):
    slide = prs.slides.add_slide(blank)
    page["n"] += 1
    box(slide, 0.6, 0.45, 12.1, 1.1, [("Backup: " if backup else "") + title], 36, color=NAVY, bold=True)
    box(slide, 0.6, 7.0, 6, 0.3, ["Group 3  |  Applied ML  |  UCI Adult"], 11, color=GREY)
    box(slide, 11.7, 7.0, 1.0, 0.3, [str(page["n"] + 1)], 11, color=GREY, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def with_figure(slide, bullets, figure):
    box(slide, 0.6, 1.8, 5.7, 4.9, bullets, 22, bullet=True, space=16)
    card(slide, 6.7, 1.6, 6.0, 5.2)
    picture(slide, figure, 6.8, 1.7, 5.8, 5.0)


def callout(slide, left, top, width, number, label, color=BLUE):
    box(slide, left, top, width, 0.9, [number], 48, color=color, bold=True)
    box(slide, left, top + 0.95, width, 0.6, [label], 16, color=GREY)


# ---- slides -----------------------------------------------------------------------------------
title = prs.slides.add_slide(blank)
title.background.fill.solid()
title.background.fill.fore_color.rgb = rgb(NAVY)
box(title, 0.9, 2.2, 11.5, 1.4, ["Predicting income above 50K"], 48, color="FFFFFF", bold=True)
box(title, 0.9, 3.5, 11.5, 0.8, ["A calibrated and fairness-audited model on the UCI Adult data"], 24, color="CFE3F3")
box(title, 0.9, 5.3, 11.5, 1.2, ["Applied ML: Foundations, Engineering & Evaluation, group 3",
                                 "Josha\u00ef Bensoussan, Lucas Noury, Elie Maarek"], 18, color="FFFFFF")

s = new_slide("Goal: predict who earns more than 50K",
              "Speaker Joshaï. What: a bank wants to offer a premium product without asking for a payslip, so we predict income from socio-demographic data "
              "(UCI Adult). Why: three questions: is the ranking good, can we trust the probabilities, is the model fair? The roadmap shows the steps "
              "and the requirement number of each one. What could invalidate it: a census sample from 1994 is not a bank's customers. "
              "Likely question: why this dataset? It is the one chosen by the group; it has a sensitive attribute (sex) for the fairness audit.")
box(s, 0.6, 1.7, 12.1, 2.2, ["A bank wants to offer a premium product to people who earn more than 50K, without a payslip",
                             f"UCI Adult: {N['nClean']} people, 14 variables, {N['posRate']} above 50K",
                             "Three questions: is the ranking good, can we trust the probabilities, is it fair?"],
    22, bullet=True, space=14)
steps = [("1", "Data and split"), ("2", "Features"), ("3", "Imbalance"), ("4-5", "3 models, 1 Pipeline"),
         ("6-7", "Optuna and run log"), ("8", "SHAP"), ("9", "Calibration"), ("10", "Fairness")]
width, gap = 1.42, 0.105
for i, (num, text) in enumerate(steps):
    left = 0.6 + i * (width + gap)
    card(s, left, 4.35, width, 1.75)
    box(s, left + 0.12, 4.5, width - 0.24, 0.5, [num], 22, color=BLUE, bold=True)
    box(s, left + 0.12, 5.1, width - 0.24, 0.9, [text], 14, color=TEXT, bold=True)
box(s, 0.6, 6.3, 12.1, 0.4, ["Requirement numbers of the project brief. At the end the model is served by an API (POST /predict)."],
    14, color=GREY)

s = new_slide("Clean data and an honest split",
              "Speaker Lucas. What: 52 exact duplicates removed before the split, stratified 60/20/20 split with seed 42, test set used once. "
              "Why: the same person in train and test would inflate the scores. What could invalidate it: no date and no person id, so we cannot "
              "make a time or group split; rows are assumed independent. Metric: PR-AUC, because three people out of four are at most 50K and "
              "accuracy hides the minority class. Likely question: what is the baseline? Always predicting the prevalence: PR-AUC 0.239.")
with_figure(s, [f"{N['nClean']} people after removing {N['nDup']} duplicates, {N['posRate']} above 50K",
                f"Stratified split {N['nTrain']} / {N['nVal']} / {N['nTest']}, test used once",
                f"Main metric: PR-AUC. Always answering no gives {N['cvDummyNone']}",
                "No date or ID: rows are assumed independent"], "slide_eda.png")

s = new_slide("New variables, no resampling",
              "Speaker Lucas. What: missing as a category, five new variables, three remedies compared with the same CV. "
              "Why: remedies must be justified by results. What could invalidate it: SMOTE interpolates one-hot columns, we did not test SMOTENC. "
              "Likely question: why not use class weights? They make the probabilities over-confident.")
with_figure(s, [f"Missing occupation is informative: {N['posRateMissOcc']} earn more than 50K, against {N['posRatePresentOcc']}",
                f"Five new variables help only the logistic regression ({N['ablLogRaw']} to {N['ablLogEng']})",
                "Class weights and SMOTE: no better ranking, worse probabilities",
                "We keep the true class mix and tune the threshold"], "slide_remedies.png")

s = new_slide("LightGBM ranks best",
              "Speaker Lucas. What: logistic regression, random forest, LightGBM, each one Pipeline. Why: each step adds something "
              "(non-linearity, then sequential correction). What could invalidate it: same folds for all models. "
              "Likely question: why does LightGBM beat the forest? Boosting corrects the errors step by step.")
with_figure(s, [f"No skill: PR-AUC {N['cvDummyNone']} (the prevalence)",
                f"Logistic {N['cvLogNone']}, forest {N['cvRfNone']}, LightGBM {N['cvLgbNone']}",
                "Trees can capture interactions that a linear model cannot",
                "One Pipeline per model, refitted in each fold"], "slide_models.png")

s = new_slide("Tuning gives a small gain",
              "Speaker Elie. What: one Optuna study per model, 5-fold stratified CV on the training set. Why: tune with CV, report with validation "
              "and test. What could invalidate it: the best CV score is optimistic, we did not use nested CV. "
              "Likely question: why not nested CV? It is optional in the course and we keep an untouched test set.")
with_figure(s, [f"Optuna: {N['nTrialsLog']}, {N['nTrialsRf']} and {N['nTrialsLgb']} trials, 5-fold CV",
                "The best CV score is optimistic: validation and test stay apart",
                f"Gain from tuning is small ({N['tunedGainLgb']} points for LightGBM)",
                f"{N['nRuns']} runs logged with parameters, metrics and artifact"], "slide_optuna.png")

s = new_slide("The model relies on sensible signals",
              "Speaker Elie. What: TreeExplainer on the validation set, one-hot columns summed back into variables. Why: check that the important "
              "variables make business sense. What could invalidate it: SHAP explains the model, not the causes of income; some variables appear "
              "twice (is_married and marital-status). Likely question: gain, permutation or SHAP? They answer different questions.")
with_figure(s, ["Age, marriage, education and occupation matter most",
                "Directions make sense: married, older, educated, managerial",
                "Sex has a small direct effect; marriage and relationship matter more",
                "SHAP explains the model, not the causes"], "slide_shap.png")

s = new_slide("We can trust the probabilities",
              "Speaker Elie. What: reliability diagram, Brier score and ECE on the validation set, Platt and isotonic compared. Why: a bank uses the "
              "probability to value a contact. What could invalidate it: the middle bins have few people; calibration holds for the 1994 population. "
              "Likely question: why not recalibrate? The gain is under 0.001 Brier, so we keep the simpler model.")
callout(s, 0.6, 1.8, 2.8, N["eceCalRaw"], "ECE of the model", BLUE)
callout(s, 3.5, 1.8, 2.8, N["eceCalWeights"], "ECE with class weights", ORANGE)
box(s, 0.6, 3.7, 5.7, 3.0, [f"Mean prediction {N['meanPredRaw']}, observed {N['obsRate']}",
                            "Class weights: over-confident in every bin",
                            "Platt and isotonic: no gain, so no recalibration"], 22, bullet=True, space=16)
card(s, 6.7, 1.6, 6.0, 5.2)
picture(s, "slide_calibration.png", 6.8, 1.7, 5.8, 5.0)

s = new_slide("Women are missed more often",
              "Speaker Joshaï. What: equal opportunity on sex, then one threshold per sex chosen on validation, before and after on the test set. "
              "Why: base rates differ, so equal selection rates would not be fair. What could invalidate it: using sex at decision time is disparate "
              "treatment; labels reflect 1994 pay; intervals ignore threshold estimation. Likely question: why not just drop the sex column? Proxies.")
callout(s, 0.6, 1.8, 2.8, N["gapPointsBefore"], "points of recall gap", ORANGE)
callout(s, 3.5, 1.8, 2.8, N["gapPointsAfter"], "after one threshold per sex", GREEN)
box(s, 0.6, 3.7, 5.7, 3.2, ["Goal: same recall for eligible women and men (equal opportunity)",
                            f"Removing the sex column helps only a little (gap {N['gapNoSex']})",
                            "The gap interval now contains zero",
                            f"Cost: accuracy {N['accBefore']} to {N['accAfter']}, lower precision for women"], 19, bullet=True, space=10)
card(s, 6.7, 1.6, 6.0, 5.2)
picture(s, "slide_fairness.png", 6.8, 1.7, 5.8, 5.0)

s = new_slide("Results, lessons and limits",
              "Speaker Joshaï. What: final model on the test set used once, with bootstrap intervals. Lessons: the model family mattered more than tuning; "
              "threshold tuning gives the same F1 as reweighting and keeps the probabilities; dropping sex is not enough. "
              "What could invalidate it: one split and one seed, 1994 data, rows assumed independent. Contribution and AI use: say who did what and how "
              "the AI assistant was used (see the report). Likely question: what next? Real costs for the threshold, monitoring, refit on train plus validation.")
callout(s, 0.6, 1.6, 3.8, N["testPrAuc"], f"PR-AUC on test (no skill {N['prevalenceTest']})")
callout(s, 4.7, 1.6, 3.8, N["testFOne"], "F1 on test")
callout(s, 8.8, 1.6, 3.8, N["brierSkill"], "Brier skill score")
box(s, 0.6, 3.75, 12.1, 3.2, ["Lessons: the model family mattered more than tuning, and threshold tuning beat resampling",
                              "Ready to use: one saved Pipeline served by an API (POST /predict)",
                              "Limits: 1994 data, one split, rows assumed independent, binary sex; using sex at decision time may be illegal",
                              "AI use: for the analysis, an AI coding assistant helped with the code and the visualisations (see the report)"],
    19, bullet=True, space=9)


# ---- backup slides ----------------------------------------------------------------------------
def table_slide(title_text, notes, rows, widths, top=1.8, size=14):
    slide = new_slide(title_text, notes, backup=True)
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(0.6), Inches(top), Inches(sum(widths)),
                                   Inches(0.4 * len(rows)))
    table = shape.table
    for j, w in enumerate(widths):
        table.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if i == 0 else ("FFFFFF" if i % 2 else LIGHT))
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.size = Pt(size)
            run.font.name = FONT
            run.font.bold = i == 0
            run.font.color.rgb = rgb("FFFFFF" if i == 0 else TEXT)
    return slide


label = {"logreg": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM", "dummy": "No skill"}
rows = [["Model", "Imbalance", "CV PR-AUC", "Val ROC-AUC", "Val F1 at 0.5", "Val F1 tuned", "Val Brier"]]
for _, r in models.iterrows():
    rows.append([label[r.model], r.imbalance, f"{r.cv_pr_auc_mean:.3f}", f"{r.val_roc_auc:.3f}",
                 f"{r['val_f1_at_0.5']:.3f}", f"{r.val_f1_tuned:.3f}", f"{r.val_brier:.3f}"])
table_slide("All models and remedies",
            "Use if asked about the full comparison. Same CV for all models; threshold from out-of-fold predictions.",
            rows, [3.0, 1.8, 1.5, 1.6, 1.6, 1.5, 1.3], size=13)

for part, pair in enumerate([("true_positive", "false_negative"), ("false_positive", "borderline")]):
    s = new_slide(f"Individual explanations ({part + 1} of 2)",
                  "Use if asked for local explanations: most confident true positive, typical false negative, typical false positive, person at the threshold.",
                  backup=True)
    for k, name in enumerate(pair):
        picture(s, f"shap_waterfall_{name}.png", 0.6 + k * 6.2, 1.7, 6.0, 4.9)

fair_ci = json.loads((TAB / "fairness_ci.json").read_text())
rows = [["Policy", "Women TPR", "Men TPR", "TPR gap [95% interval]", "FPR gap", "DP ratio"]]
summary = pd.read_csv(TAB / "fairness_summary_test.csv").set_index("policy")
key = {"global threshold": "before", "no sex column": "no_sex_column", "group thresholds": "after"}
tag = {"global threshold": "Before", "no sex column": "NoSex", "group thresholds": "After"}
for policy, r in summary.iterrows():
    lo, hi = fair_ci[key[policy]]["eq_opp_gap_tpr"]
    rows.append([policy.capitalize(), N["tprFemale" + tag[policy]], N["tprMale" + tag[policy]],
                 f"{r.eq_opp_gap_tpr:.3f} [{lo:.3f}, {hi:.3f}]", f"{r.fpr_gap:.3f}", f"{r.dp_ratio:.2f}"])
slide = table_slide("Fairness audit in detail",
                    "Use if asked about the audit. Thresholds per sex come from the validation set; the test set is only used for the comparison.",
                    rows, [3.0, 1.7, 1.7, 3.2, 1.3, 1.2], size=14)
box(slide, 0.6, 4.2, 12.1, 2.4,
    [f"Group thresholds: women {N['thrFemaleAfter']}, men {N['thrMaleAfter']} (target recall {N['targetTpr']} from validation)",
     f"{N['extraWomenContacted']} more women contacted, {N['extraWomenEligible']} of them eligible ({N['extraWomenShare']})",
     f"Run log: {N['nRuns']} runs, of which {N['nOptunaRuns']} Optuna trials"], 20, bullet=True, space=12)

prs.save(OUT / "presentation_group3.pptx")
print("saved", OUT / "presentation_group3.pptx", "with", len(prs.slides), "slides")
