from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5
N_TRIALS = {"logreg": 30, "random_forest": 30, "lightgbm": 60}
SENSITIVE = "sex"

FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
RUNS = ROOT / "runs"
MODELS = RUNS / "models"

for folder in (FIG, TAB, RUNS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)
