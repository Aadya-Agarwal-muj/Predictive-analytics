"""
config.py - one place for every path and constant used by the project.

Why a separate config file?
    Every script and the dashboard need the same folder locations and the same
    column names. Defining them once avoids copy-paste mistakes, and because
    all paths are built from PROJECT_ROOT (not from C:\\Users\\...), the code
    runs unchanged on any computer that clones the GitHub repository.
"""

from pathlib import Path

# Path(__file__) is this file: <project>/src/config.py
# .resolve() turns it into an absolute path; .parents[1] goes up two levels
# (src -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARCHIVE_NAME = "SMPracticals_1.4-3.2.tar.gz"
ARCHIVE_PATH = RAW_DIR / ARCHIVE_NAME                 # original CRAN package
ARCHIVE_MEMBER = "SMPracticals/data/limits.rda"       # file inside the archive
RDA_PATH = RAW_DIR / "limits.rda"                     # extracted R data file
RAW_CSV_PATH = RAW_DIR / "limits_raw.csv"             # faithful CSV copy
PROCESSED_CSV_PATH = PROCESSED_DIR / "swedish_speed_limit_92.csv"

# -------------------------------------------------------------- output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_results"

# ------------------------------------------------------- dataset definitions
EXPECTED_ROWS = 92
EXPECTED_COLUMNS = ["day", "lim1", "lim2", "y1", "y2"]
BINARY_COLUMNS = ["lim1", "lim2"]
COUNT_COLUMNS = ["y1", "y2"]

TARGET = "y2"                              # 1962 accident count (what we predict)
FEATURES = ["y1", "lim1", "lim2"]          # final predictor set
ID_COLUMN = "day"                          # identifier, NOT a predictor

# Alternative predictor sets compared during feature selection (script 06).
FEATURE_SETS = {
    "y1 + lim1 + lim2 (main)": ["y1", "lim1", "lim2"],
    "y1 + lim2": ["y1", "lim2"],
    "log_y1 + lim1 + lim2": ["log_y1", "lim1", "lim2"],
    "lim1 + lim2 (no history)": ["lim1", "lim2"],
    # Extension motivated by EDA: counts show a strong lag-7 autocorrelation
    # along the day code, suggesting consecutive days with a weekly cycle.
    # "day_mod7" is expanded into 6 indicator columns by design_matrix().
    "y1 + lim1 + lim2 + weekly cycle": ["y1", "lim1", "lim2", "day_mod7"],
}

RANDOM_STATE = 42   # fixed seed so every split is reproducible


def ensure_output_dirs() -> None:
    """Create output folders if they do not exist (safe to call repeatedly)."""
    for folder in (PROCESSED_DIR, FIG_DIR, METRICS_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)
