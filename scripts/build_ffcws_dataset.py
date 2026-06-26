
#!/usr/bin/env python3
"""
Build an analysis-ready parental-mental-health panel from the FFCWS public-use file.

Narrative 1 (parental mental health). Produces:
  data/processed/ffcws_parental_mh_long.csv   one row per family x wave x parent
  data/processed/ffcws_parental_mh_wide.csv   one row per family (raw selected vars)
  data/processed/ffcws_build_summary.txt      QA summary (N, prevalence, missingness)

USAGE:
  1. Download the FFCWS public-use merged file from ICPSR/DSDR study 31622
     (free Researcher Passport account + accept Terms of Use; NO approval needed).
     Choose Stata (.dta) or Delimited (.csv) format.
  2. Put the data file in  data/raw/  (any .dta or .csv; the merged "all-waves" file).
  3. Run:  python3 scripts/build_ffcws_dataset.py

Variable names verified against the public wave codebooks (see FFCWS_variable_map.md).
"""
import sys, glob, os
import pandas as pd
import numpy as np
from paths import DATA, RAW, DTA, PROCESSED as OUT

# wave number -> human label (child age). CIDI-SF caseness fielded Years 1-9.
# Year 15 (wave 6, cp6*) depression is NOT in the public-use file (metadata in_FFC_file=No), so excluded.
WAVES = {2: "Year 1", 3: "Year 3", 4: "Year 5", 5: "Year 9"}
# both parents (cm/cf) present at all included waves
PARENT_PREFIX = {"mother": "cm", "father": "cf", "pcg": "cp"}
GAD_WAVES = {2, 3}  # GAD caseness only fielded at Years 1 & 3

# baseline covariates (constant per family)
HH_COVARS = ["cm1hhinc", "cm1povca", "cm1relf", "cm1bsex"]          # household-level
MOM_COVARS = {"age": "cm1age", "edu": "cm1edu", "race": "cm1ethrace"}
DAD_COVARS = {"age": "cf1age", "edu": "cf1edu", "race": "cf1ethrace"}
PRIORMH = {"mother": "m2j12", "father": "f2j12"}  # CIDI stem, Year 1 (earliest proxy)
WEIGHTS = ["m1natwt", "f1natwt", "m1citywt", "f1citywt"]


def find_input():
    # Prefer the ICPSR "All Years" merged file (DS0001); else any data .dta/.csv.
    cands = glob.glob(str(DTA.parent / "*-Data.dta"))
    if cands:
        return max(cands, key=os.path.getsize)
    for d in (RAW, DATA):
        for ext in ("*.dta", "*.csv"):
            hits = [f for f in glob.glob(os.path.join(d, ext))
                    if "cb" not in os.path.basename(f).lower()
                    and "metadata" not in os.path.basename(f).lower()]
            if hits:
                return max(hits, key=os.path.getsize)
    return None


def load_selected(path, desired_lower):
    """Read ONLY the target columns (case-insensitive). ICPSR stores names UPPERCASE."""
    print(f"Loading {path}\n  selecting {len(desired_lower)} target columns ...")
    if path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df[[c for c in desired_lower if c in df.columns]]
    # .dta: read 1-row header to map case, then read only needed columns
    hdr = pd.read_stata(path, iterator=True).read(1)
    lower2actual = {c.lower(): c for c in hdr.columns}
    actual = [lower2actual[c] for c in desired_lower if c in lower2actual]
    df = pd.read_stata(path, columns=actual, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def clean_missing(s, valid01=False):
    """FFCWS negatives (-1..-9) are missing codes -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s >= 0, np.nan)
    if valid01:
        s = s.where(s.isin([0, 1]), np.nan)
    return s


def target_columns():
    cols = ["idnum"] + HH_COVARS + list(MOM_COVARS.values()) + list(DAD_COVARS.values())
    cols += list(PRIORMH.values()) + WEIGHTS
    for w in WAVES:
        for parent in (["mother", "father"] if w < 6 else ["pcg"]):
            p = PARENT_PREFIX[parent]
            cols += [f"{p}{w}md_case_lib", f"{p}{w}md_case_con"]
            if w in GAD_WAVES:
                cols += [f"{p}{w}gad_case"]
    return cols


def build_long(df):
    rows = []
    for w, label in WAVES.items():
        for parent in (["mother", "father"] if w < 6 else ["pcg"]):
            p = PARENT_PREFIX[parent]
            lib, con = f"{p}{w}md_case_lib", f"{p}{w}md_case_con"
            if lib not in df.columns:
                print(f"  [skip] {lib} not in file"); continue
            cov = MOM_COVARS if parent in ("mother", "pcg") else DAD_COVARS
            block = pd.DataFrame({
                "idnum": df["idnum"],
                "wave": w,
                "child_age": label,
                "parent": parent,
                "md_case_lib": clean_missing(df[lib], valid01=True),
                "md_case_con": clean_missing(df.get(con), valid01=True),
                "gad_case": clean_missing(df[f"{p}{w}gad_case"], valid01=True)
                            if f"{p}{w}gad_case" in df.columns else np.nan,
                "par_age": clean_missing(df.get(cov["age"])),
                "par_edu": clean_missing(df.get(cov["edu"])),
                "par_race": clean_missing(df.get(cov["race"])),
                "hh_income": clean_missing(df.get("cm1hhinc")),
                "poverty_cat": clean_missing(df.get("cm1povca")),
                "rel_status": clean_missing(df.get("cm1relf")),
                "baby_sex": clean_missing(df.get("cm1bsex")),
            })
            rows.append(block)
    long = pd.concat(rows, ignore_index=True)
    # EXCLUSION CRITERION (defines the analytic sample): keep only parent x wave rows
    # with >=1 observed depression measure; drop rows missing BOTH md_case_lib and
    # md_case_con. A family disappears entirely only if it has ZERO depression obs across
    # every parent x wave cell -> 4898 source families -> 4740 analytic (158 fully-unmeasured
    # families excluded). Consequence: md_case_lib reads ~0% missing downstream BY
    # CONSTRUCTION -- depression non-response must be read from measured-N / attrition,
    # not a missing% column.
    long = long.dropna(subset=["md_case_lib", "md_case_con"], how="all")
    return long


def summarize(df, long):
    L = []
    L.append("FFCWS PARENTAL MENTAL-HEALTH BUILD — SUMMARY\n" + "=" * 48)
    L.append(f"Source rows (families): {len(df)}")
    tgt = target_columns()
    present = [c for c in tgt if c in df.columns]
    missing = [c for c in tgt if c not in df.columns]
    L.append(f"Target variables found: {len(present)}/{len(tgt)}")
    if missing:
        L.append(f"  MISSING (verify tier/name): {', '.join(missing)}")
    L.append("\nDepression (md_case_lib) prevalence & N by wave x parent:")
    g = (long.groupby(["wave", "child_age", "parent"])
              .agg(n=("md_case_lib", "size"),
                   n_measured=("md_case_lib", lambda s: s.notna().sum()),
                   dep_rate=("md_case_lib", "mean")).reset_index())
    for _, r in g.iterrows():
        rate = f"{r['dep_rate']*100:.1f}%" if pd.notna(r["dep_rate"]) else "NA"
        L.append(f"  w{int(r['wave'])} {r['child_age']:<8} {r['parent']:<7} "
                 f"n={int(r['n']):>5}  measured={int(r['n_measured']):>5}  dep={rate}")
    # father attrition flag
    fa = g[g.parent == "father"].sort_values("wave")
    if len(fa) > 1:
        L.append(f"\nFather measured N: Year1={int(fa.iloc[0]['n_measured'])} "
                 f"-> last={int(fa.iloc[-1]['n_measured'])} (attrition check)")
    return "\n".join(L)


def main():
    path = find_input()
    if not path:
        print(__doc__)
        print("\n>>> No FFCWS data file found in data/raw/. "
              "Download it first (see USAGE above), then re-run. <<<")
        sys.exit(0)
    df = load_selected(path, target_columns())
    if "idnum" not in df.columns:
        sys.exit("ERROR: 'idnum' not found — is this the FFCWS merged file?")

    long = build_long(df)
    wide = df[[c for c in target_columns() if c in df.columns]].copy()

    long.to_csv(f"{OUT}/ffcws_parental_mh_long.csv", index=False)
    wide.to_csv(f"{OUT}/ffcws_parental_mh_wide.csv", index=False)
    summary = summarize(df, long)
    with open(f"{OUT}/ffcws_build_summary.txt", "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nWrote:\n  {OUT}/ffcws_parental_mh_long.csv  ({len(long)} rows)"
          f"\n  {OUT}/ffcws_parental_mh_wide.csv  ({len(wide)} cols={wide.shape[1]})"
          f"\n  {OUT}/ffcws_build_summary.txt")


if __name__ == "__main__":
    main()
