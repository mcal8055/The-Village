"""
MARTS LAYER — consumption-ready tables the models read. These write the processed
CSVs the Rmd, causal scripts, and DS notebook consume: tested, canonical features.

  dim_family_vulnerability -> data/processed/preexisting_vuln.csv   (family grain)
  fct_village_panel_tv     -> data/processed/village_panel_tv.csv    (lagged-transition grain)

Caseness + covariates are consumed from data/processed/ffcws_parental_mh_long.csv, an
established upstream source (built by build_ffcws_dataset.py); only the two ENGINEERED
constructs (Village instrumental, family history) are re-derived canonically here.
"""
import numpy as np
import pandas as pd
from paths import PROCESSED as OUT

CASENESS_LONG = f"{OUT}/ffcws_parental_mh_long.csv"
TRANSITION_PAIRS = [(2, 3), (3, 4), (4, 5)]   # Y1->Y3, Y3->Y5, Y5->Y9
MODEL_WAVES = [2, 3, 4, 5]            # Y1, Y3, Y5, Y9 (all four CIDI-caseness waves)


def _wave_aggregates(stg, value_col, prefix, waves=MODEL_WAVES):
    """Per idnum x parent: the Y1 baseline value and the across-wave mean of a facet."""
    d = stg[stg.wave.isin(waves)]
    mean = d.groupby(["idnum", "parent"])[value_col].mean().reset_index(name=f"{prefix}_mean")
    y1 = (d[d.wave == 2][["idnum", "parent", value_col]]
          .rename(columns={value_col: f"{prefix}_y1"}))
    return mean.merge(y1, on=["idnum", "parent"], how="outer")


def dim_family_vulnerability(int_fh, stg_cesd):
    """Family-grain wide table. Keeps fh_mother/fh_father (= fh_any)
    and cesd_father_baseline; adds graded burden/severity/suicide per parent."""
    wide = int_fh.pivot(index="idnum", columns="parent",
                        values=["fh_any", "fh_burden", "fh_severity", "fh_suicide"])
    wide.columns = [f"{metric.replace('fh_any', 'fh')}_{parent}".replace("fh_", "fh_", 1)
                    if metric == "fh_any" else f"{metric}_{parent}"
                    for metric, parent in wide.columns]
    # tidy names: fh_any -> fh_<parent>; others -> fh_<x>_<parent>
    rename = {}
    for parent in ("mother", "father"):
        rename[f"fh_{parent}"] = f"fh_{parent}"               # from fh_any
        for x in ("burden", "severity", "suicide"):
            rename[f"fh_{x}_{parent}"] = f"fh_{parent}_{x}"
    wide = wide.rename(columns=rename).reset_index()
    out = wide.merge(stg_cesd, on="idnum", how="left")
    cols = ["idnum", "fh_mother", "fh_father", "cesd_father_baseline",
            "fh_mother_burden", "fh_father_burden",
            "fh_mother_severity", "fh_father_severity",
            "fh_mother_suicide", "fh_father_suicide"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(f"{OUT}/preexisting_vuln.csv", index=False)
    return out


def fct_village_panel_tv(int_village, stg_realized, stg_enacted, stg_grandparent, stg_religion, dim_fh):
    """Lagged-transition panel: prior-wave exposure -> next-wave CIDI caseness.

    FOUR SIBLING Village facets, never folded into one index (the multi-axis
    view of the village):
      v_perceived  (= v_instrumental) : CAPACITY;       expected protective (negative)
      v_enacted                       : MOBILIZATION;   expected POSITIVE (need-driven)
      v_grandparent                   : EMBEDDEDNESS;   structural co-residence (sign open)
      v_religion                      : PARTICIPATION;  attends services weekly+ (ablation)
    v_realized (gave-to-kin outflow) is a separate column.
    enacted_discordant splits receipt by prior depression — the cut that distinguishes
    'village showed up' (received while not previously depressed) from 'mobilization'."""
    long = pd.read_csv(CASENESS_LONG)
    fh_long = (dim_fh.melt("idnum", ["fh_mother", "fh_father"],
                           var_name="parent", value_name="fh")
               .assign(parent=lambda d: d.parent.str.replace("fh_", "", regex=False)))
    base = (long[["idnum", "wave", "parent", "md_case_lib", "md_case_con",
                  "par_edu", "poverty_cat"]]
            .merge(int_village[["idnum", "parent", "wave", "v_instrumental"]],
                   on=["idnum", "parent", "wave"], how="left")
            .merge(stg_realized, on=["idnum", "parent", "wave"], how="left")
            .merge(stg_enacted, on=["idnum", "parent", "wave"], how="left")
            .merge(stg_grandparent, on=["idnum", "parent", "wave"], how="left")
            .merge(stg_religion, on=["idnum", "parent", "wave"], how="left")
            .merge(fh_long, on=["idnum", "parent"], how="left"))

    out = []
    for wp, wt in TRANSITION_PAIRS:
        prev = base[base.wave == wp].set_index(["idnum", "parent"])
        cur = base[base.wave == wt].set_index(["idnum", "parent"])
        d = pd.DataFrame(index=cur.index)
        d["dep_t"] = cur["md_case_lib"]
        d["dep_t_con"] = cur["md_case_con"]
        d["dep_prev"] = prev["md_case_lib"]
        d["v_instrumental"] = prev["v_instrumental"]      # lagged exposure (perceived)
        d["v_perceived"] = prev["v_instrumental"]         # explicit perceived-axis alias
        d["v_realized"] = prev["v_realized"]              # gave-to-kin (outflow)
        d["v_enacted"] = prev["v_enacted"]                # received/mobilized (expected +)
        d["v_grandparent"] = prev["v_grandparent"]        # structural embeddedness (sign open)
        d["v_religion"] = prev["v_religion"]              # participation: weekly+ attendance
        # discordance: among receivers only, 1 = received while NOT previously depressed
        # (closer to exogenous village activity), 0 = received while depressed (mobilization).
        recv = prev["v_enacted"].reindex(cur.index)        # align to outcome-wave grain
        depp = prev["md_case_lib"].reindex(cur.index)
        d["enacted_discordant"] = np.where(
            recv.to_numpy() == 1, (depp.to_numpy() == 0).astype(float), np.nan)
        d["fh"] = prev["fh"]
        d["poverty_cat"] = prev["poverty_cat"]
        d["par_edu"] = prev["par_edu"]
        d["wave_out"] = wt
        out.append(d.reset_index())
    panel = pd.concat(out, ignore_index=True).dropna(subset=["dep_t", "dep_prev"])
    panel.to_csv(f"{OUT}/village_panel_tv.csv", index=False)
    return panel


def fct_village_wave(int_village, stg_realized, stg_enacted, stg_grandparent, stg_religion):
    """Wave-level (un-lagged) village facets: one row per idnum x parent x wave. The causal
    Rmd reads this to assemble its `base` / reverse-lag tables. Same scope as the panel."""
    keys = ["idnum", "parent", "wave"]
    v = int_village[keys + ["v_instrumental"]].copy()
    v["v_perceived"] = v["v_instrumental"]
    out = (v.merge(stg_realized, on=keys, how="outer")
           .merge(stg_enacted, on=keys, how="outer")
           .merge(stg_grandparent, on=keys, how="outer")
           .merge(stg_religion, on=keys, how="outer"))
    out = out[out.wave.isin(MODEL_WAVES)].sort_values(keys).reset_index(drop=True)
    out.to_csv(f"{OUT}/village_wave.csv", index=False)
    return out


def obt_model_person(int_village, stg_realized, stg_enacted, stg_grandparent, stg_religion, dim_fh):
    """One-big-table person-level modeling mart (one row per family x parent) — the single
    source the PREDICTIVE pipeline reads. Consolidates everything ds_build_profile.py derived.
    Persistence honors the Y5 cap (Y1/Y3/Y5).

    Emits data/processed/obt_model_person.csv AND ds_person.csv (drop-in for the notebook)."""
    long = pd.read_csv(CASENESS_LONG)
    long = long[long.wave.isin(MODEL_WAVES)].copy()

    # ---- caseness wide + persistence targets ----
    cw = long.pivot_table(index=["idnum", "parent"], columns="wave", values="md_case_lib")
    cw.columns = [f"case_w{c}" for c in cw.columns]
    cw = cw.reset_index()
    nobs = (long.groupby(["idnum", "parent"])["md_case_lib"]
            .agg(["count", "sum", "mean"]).reset_index())
    nobs.columns = ["idnum", "parent", "n_waves", "n_cases", "case_rate"]
    person = nobs.merge(cw, on=["idnum", "parent"], how="left")

    # ---- village facets: Y1 baseline + across-wave mean (perceived=support_*) ----
    vill = int_village.rename(columns={"v_instrumental": "v_perceived"})
    person = (person
              .merge(_wave_aggregates(vill, "v_perceived", "support"), on=["idnum", "parent"], how="left")
              .merge(_wave_aggregates(stg_enacted, "v_enacted", "enacted"), on=["idnum", "parent"], how="left")
              .merge(_wave_aggregates(stg_grandparent, "v_grandparent", "grandparent"), on=["idnum", "parent"], how="left")
              .merge(_wave_aggregates(stg_religion, "v_religion", "religion"), on=["idnum", "parent"], how="left")
              .merge(stg_realized[stg_realized.wave.isin(MODEL_WAVES)]
                     .groupby(["idnum", "parent"])["v_realized"].mean().reset_index(name="realized_mean"),
                     on=["idnum", "parent"], how="left"))

    # ---- vulnerability: graded family history (per parent) + father baseline CES-D ----
    fh_long = (dim_fh.melt("idnum",
               ["fh_mother", "fh_father", "fh_mother_burden", "fh_father_burden",
                "fh_mother_severity", "fh_father_severity", "fh_mother_suicide", "fh_father_suicide"],
               var_name="col", value_name="val"))
    fh_long["parent"] = fh_long["col"].str.extract(r"fh_(mother|father)")
    fh_long["metric"] = (fh_long["col"].str.replace(r"fh_(mother|father)_?", "", regex=True)
                         .replace("", "fh"))
    fh_wide = fh_long.pivot_table(index=["idnum", "parent"], columns="metric", values="val").reset_index()
    fh_wide = fh_wide.rename(columns={"burden": "fh_burden", "severity": "fh_severity",
                                      "suicide": "fh_suicide"})
    person = person.merge(fh_wide, on=["idnum", "parent"], how="left")
    person = person.merge(dim_fh[["idnum", "cesd_father_baseline"]], on="idnum", how="left")
    person.loc[person.parent == "mother", "cesd_father_baseline"] = np.nan

    # ---- SES / demographics / baseline anxiety (from caseness long) ----
    ses = (long.groupby(["idnum", "parent"])[["poverty_cat", "par_edu", "par_age",
                                              "hh_income", "par_race"]].first().reset_index()
           .rename(columns={"hh_income": "hhinc", "par_race": "ethrace"}))
    person = person.merge(ses, on=["idnum", "parent"], how="left")
    gad1 = (long[long.wave == 2][["idnum", "parent", "gad_case"]]
            .rename(columns={"gad_case": "gad_y1"}))
    person = person.merge(gad1, on=["idnum", "parent"], how="left")

    # ---- targets; require >=2 observed waves to define persistence ----
    person["persistent"] = (person["n_cases"] >= 2).astype(int)
    person["persistent_majority"] = (person["case_rate"] >= 0.5).astype(int)
    person = person[person["n_waves"] >= 2].reset_index(drop=True)

    person.to_csv(f"{OUT}/obt_model_person.csv", index=False)
    person.to_csv(f"{OUT}/ds_person.csv", index=False)        # drop-in for the predictive notebook
    return person
