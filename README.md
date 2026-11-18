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

## Protocol

* The 52 exact duplicates are removed before the split.
* Stratified split: 60% train, 20% validation, 20% test (seed 42).
* Train: fitting, cross-validation, Optuna, choice of the model family.
  Validation: calibration method, decision threshold, fairness thresholds.
  Test: used once, in `06_final_test.py`, then for the fairness comparison.
* All preprocessing is inside the `Pipeline`, so it is refitted on the training folds only.

## Use of AI

We used an AI coding assistant to write a first version of the code and of the report,
then to simplify the code and to rewrite the report. See the section on AI use in the report
for what the group did and checked itself.
