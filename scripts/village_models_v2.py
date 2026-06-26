"""
v2 Village modeling: WITHIN-PERSON identification of "does support mitigate depression".

Outcome: binary CIDI md_case_lib at wave t (dep_t).
Exposure: v_instrumental measured at PRIOR wave, decomposed (Mundlak / within-between):
   v_ib = person-mean support  (BETWEEN-person term; confounded by stable disposition)
   v_iw = wave deviation        (WITHIN-person term;  the "mitigation" estimate)
fh (family-history vulnerability) = between-person axis; fh x v_iw = buffering test.

Two estimators (convergence check):
  (1) GEE logit, Mundlak-decomposed, cluster-robust  -> between & within + buffering
  (2) Conditional fixed-effects logit (pure within)  -> robustness on v_iw / fh x v_iw
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from paths import PROCESSED

P = pd.read_csv(PROCESSED/"village_panel_tv.csv")

def z(s): return (s - s.mean()) / s.std()

for parent in ["mother", "father"]:
    d = P[P.parent == parent][["idnum","dep_t","dep_prev","wave_out",
                               "v_instrumental","fh","poverty_cat","par_edu"]].dropna().copy()
    # standardize continuous predictors on the analytic sample
    d["v_instrumental"] = z(d["v_instrumental"])
    d["poverty_cat"] = z(d["poverty_cat"]); d["par_edu"] = z(d["par_edu"])
    # Mundlak decomposition of the (standardized) exposure
    d["v_ib"] = d.groupby("idnum")["v_instrumental"].transform("mean")     # between
    d["v_iw"] = d["v_instrumental"] - d["v_ib"]                            # within
    n_switch = (d.groupby("idnum")["v_iw"].transform(lambda s: s.abs().sum()) > 1e-9)
    print(f"\n############## {parent.upper()}  (rows={len(d)}, fams={d.idnum.nunique()}, "
          f"within-informative rows={int(n_switch.sum())}) ##############")

    # ---- (1) GEE logit, Mundlak-decomposed ----
    f = ("dep_t ~ dep_prev + C(wave_out) + v_ib + v_iw + fh + poverty_cat + par_edu + v_iw:fh")
    g = smf.gee(f, "idnum", data=d, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    OR, CI, pv = np.exp(g.params), np.exp(g.conf_int()), g.pvalues
    print("  -- GEE (OR [95% CI], cluster-robust) --")
    for t in ["v_ib","v_iw","v_iw:fh","fh","dep_prev"]:
        if t in OR.index:
            s = "*" if pv[t] < .05 else " "
            lab = {"v_ib":"support BETWEEN","v_iw":"support WITHIN (mitigation)",
                   "v_iw:fh":"WITHIN x fh (buffering)","fh":"family history",
                   "dep_prev":"prior depression"}[t]
            print(f"   {lab:30} OR {OR[t]:.3f} [{CI.loc[t,0]:.3f},{CI.loc[t,1]:.3f}] p={pv[t]:.3f}{s}")

    # ---- (2) Conditional FE logit (pure within) ----
    d["v_iw_fh"] = d["v_iw"] * d["fh"]
    exog = d[["dep_prev","v_iw","v_iw_fh"]].copy()
    for w in sorted(d.wave_out.unique())[1:]:
        exog[f"wave_{w}"] = (d.wave_out == w).astype(float)
    try:
        cl = ConditionalLogit(d["dep_t"].values, exog.values, groups=d["idnum"].values).fit(disp=0)
        names = list(exog.columns)
        clOR, clCI, clp = np.exp(cl.params), np.exp(cl.conf_int()), cl.pvalues
        print("  -- Conditional FE logit (pure within; switchers only) --")
        for i, nm in enumerate(names):
            if nm in ("v_iw","v_iw_fh","dep_prev"):
                s = "*" if clp[i] < .05 else " "
                lab = {"v_iw":"support WITHIN (mitigation)","v_iw_fh":"WITHIN x fh (buffering)",
                       "dep_prev":"prior depression"}[nm]
                print(f"   {lab:30} OR {clOR[i]:.3f} [{clCI[i,0]:.3f},{clCI[i,1]:.3f}] p={clp[i]:.3f}{s}")
    except Exception as e:
        print("  -- Conditional FE logit failed:", e)
