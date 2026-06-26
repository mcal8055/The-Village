"""Canonical filesystem anchors for the SSPP / FFCWS project.

Every path here is ABSOLUTE, derived from this file's own location, so scripts
resolve their data and output locations identically no matter which working
directory they are launched from — PyCharm's "run current file" (cwd = the
script's folder), `python -m scripts.foo` from the repo root, `pytest`, or cron.

Use these instead of hardcoding cwd-relative strings:

    from paths import PROCESSED, FIGURES, DTA
    df = pd.read_csv(PROCESSED / "ffcws_parental_mh_long.csv")
    fig.savefig(FIGURES / "eda_prevalence.png")

For `import paths` to resolve, the repo root must be on sys.path. That holds for
`python -m ...` run from the root, for pytest (via the sibling conftest.py), and
in PyCharm once the project root is marked as a Sources Root.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- inputs ---
DATA     = ROOT / "data"
RAW      = DATA / "raw"
DTA      = DATA / "ICPSR_31622" / "DS0001" / "31622-0001-Data.dta"
METADATA = DATA / "FFMetadata_v20_f.csv"

# --- outputs ---
PROCESSED       = DATA / "processed"
CATALOG         = DATA / "catalog"
FIGURES         = ROOT / "figures"
FIGURES_REPRO   = FIGURES / "repro"
ANALYSIS        = ROOT / "analysis"
ANALYSIS_STAGE1 = ANALYSIS / "stage1"
ANALYSIS_REPRO  = ANALYSIS / "repro"

# Path to the data-scientist skill scripts invoked via subprocess in the repro layer.
SKILLS = ROOT / ".claude" / "skills"

# Make sure output directories exist (harmless if already present). Centralized
# here so individual scripts no longer need their own os.makedirs calls.
for _d in (PROCESSED, CATALOG, FIGURES, FIGURES_REPRO, ANALYSIS_STAGE1, ANALYSIS_REPRO):
    _d.mkdir(parents=True, exist_ok=True)
