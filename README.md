# Predicting income above 50K on the UCI Adult dataset

Applied ML: Foundations, Engineering & Evaluation, final project of group 3
(Joshaï Bensoussan, Lucas Noury, Elie Maarek).

Dataset: UCI Adult (census income, 48,842 people, 1994), loaded with
`fetch_openml(name="adult", version=2)`. If OpenML cannot be reached, the script reads
`data/raw/adult-all.csv`, which contains the same rows. We checked that both sources give exactly the same split.
Target: income above 50K (23.9% of the people). Sensitive attribute: sex.

## Run the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

The full run takes between 15 and 40 minutes depending on the machine, mostly because of Optuna.
Everything is seeded (`SEED = 42` in `config.py`).

## Files

| File | What it does | Requirement |
|---|---|---|
| `01_eda.py` | Missing values, group differences, split sizes (training set only) | 1 |
| `02_compare_models.py` | No-skill baseline, 3 models, 3 imbalance strategies, feature ablation | 2, 3, 4, 5 |
| `03_tune_optuna.py` | One Optuna study per model, 5-fold CV, choice of the final family | 6, 7 |
| `04_explain.py` | SHAP (global and 4 individual cases) and permutation importance | 8 |
| `05_calibration.py` | Reliability diagram, Brier, ECE, Platt and isotonic, final model and threshold | 9 |
| `06_final_test.py` | Evaluation on the test set (once) with bootstrap confidence intervals | 4, 6 |
| `07_fairness.py` | Audit by sex, group thresholds, comparison before and after | 10 |
| `common.py` | Data loading, split, features, the Pipeline, metrics, run log | |
| `config.py` | Seed, number of folds, number of Optuna trials, folders | |

Report and slides:

| File | What it does |
|---|---|
| `make_report_tables.py` | Writes the tables and the numbers of the report from `reports/tables` (`report/generated`) |
| `check_consistency.py` | Checks the sums, the logic of the results and the claims of the report against the tables |
| `make_slides.py` | Builds `slides/presentation_group3.pptx` |
| `serve.py` | Small FastAPI app that serves the final model |

Run the tests:

```bash
pytest
```

Serve the model:

```bash
uvicorn serve:app
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"age": 32, "workclass": "Federal-gov", "education-num": 15, "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty", "relationship": "Husband", "race": "White", "sex": "Male", "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States"}'
```

The answer has the probability of earning more than 50K and the decision at the saved threshold.
`runs/final_model.joblib` and `runs/final_model.json` are versioned so that this works after cloning.
The other fitted models in `runs/models` are not versioned and are rebuilt by `run_all.sh`.

Compile the report with `cd report && tectonic report_group3.tex` (or any LaTeX engine).

`runs/tuning_log.csv` lists the Optuna trials only (parameters, fold scores, artifact).

Outputs: `reports/tables` and `reports/figures` (used in the report), `runs/run_log.csv`
(one row per run with its parameters, metrics and artifact), `runs/models` (fitted pipelines,
not versioned). The report is in `report/`.

## Main results

Final model: tuned LightGBM, probabilities kept as they are, decision threshold 0.34.
Test set, used once (PR-AUC with a 95% bootstrap interval):

| Metric | Value |
|---|---|
| PR-AUC | 0.837 [0.825, 0.848] |
| ROC-AUC | 0.932 |
| F1 / precision / recall | 0.738 / 0.686 / 0.798 |
| Brier score / ECE | 0.086 / 0.011 |

Fairness by sex (equal opportunity): the recall gap between men and women falls from
11.0 to 2.9 points with one threshold per sex, for a drop of 0.3 points of accuracy.

## Protocol

* The 52 exact duplicates are removed before the split.
* Stratified split: 60% train, 20% validation, 20% test (seed 42).
* Train: fitting, cross-validation, Optuna, choice of the model family.
  Validation: calibration method, decision threshold, fairness thresholds.
  Test: used once, in `06_final_test.py`, then for the fairness comparison.
* All preprocessing is inside the `Pipeline`, so it is refitted on the training folds only.

## Limits

* The data are from 1994 and have no date or person id, so we could not test a time split or a group split.
* The best CV score of each Optuna study is slightly optimistic. We report validation and test scores, and we did not run a nested CV.
* Sex is a variable of the model and the group thresholds use it at decision time. This may not be allowed in a real setting.
* The false positive rate gap between men and women is not removed, so equalised odds is not reached.

## Use of AI

For the analysis, we used an AI coding assistant to help us with the code and the visualisations.
We chose the topic and the dataset, took the final decisions and checked the results against the result tables.
See the section on AI use in the report.

## Report structure

The report (`report/report_group3.pdf`) has 9 pages of main text: business problem and data (1), exploratory analysis and split (2), features (3), imbalance and models (4), tuning and run log (5), explainability (6), calibration (7), fairness (8) and conclusion (9). The contribution statement, the AI-use statement and the appendices come after.

## Team

Group 3: Joshaï Bensoussan, Lucas Noury, Elie Maarek. Lucas worked mostly on the analysis (data exploration, models, tuning, explanations, calibration, fairness audit). Joshaï worked mostly on the report (creation, structure, format). Elie helped on the analysis and on checking the results. The full statement is in the report.

## Quick check

After `pip install -r requirements.txt`, two commands check the project in less than a minute: `pytest` runs the 10 tests (split, pipeline, metrics, API) and `python check_consistency.py` checks the sums and the claims of the report against the result tables. `./run_all.sh` rebuilds all the results from scratch.

## Notes for running the code

* We ran everything with Python 3.12.
* If OpenML cannot be reached, the data are read from `data/raw/adult-all.csv` (same rows, same split).
* The full run (`./run_all.sh`) takes 15 to 40 minutes, mostly because of Optuna. The tests and the consistency check are much faster.
