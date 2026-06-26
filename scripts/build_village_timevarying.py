"""
Build the TIME-VARYING Village panel for within-person / within-between models.

Change vs v1 (village_panel.csv): the village exposure is no longer fixed at Year 1.
v_instrumental (perceived instrumental support) and v_realized (gave/loaned to kin)
are measured at EACH wave, then entered LAGGED (prior wave) to predict next-wave
CIDI caseness -- the "support -> subsequent depression" alignment.

Outcome stays binary CIDI md_case_lib (only depression measure repeated all 4 waves).
fh (family-history vulnerability) and covariates are reused from existing processed files.

Wave codes: 2=Y1, 3=Y3, 4=Y5, 5=Y9.
Instrumental battery names SHIFT across waves (verified against DS0001):
  Y1 g6/g8 (NOT h3 -> that is a $-amount follow-up), Y3/Y5 h3-h6, Y9 e3-e6. 1=yes,2=no.
Realized help m#l2/f#l2 (Y3,Y5) and m5j2/f5j2 (Y9); 1=yes,2=no; none at Y1.
NOTE on "childcare" -- two DIFFERENT variables share the word; do not confuse them:
  INCLUDED: perceived emergency childcare = battery item #3 (*g6c/*g8c Y1, *h5 Y3/Y5,
    *e5 Y9). A yes/no PERCEIVED-AVAILABILITY item ("could you count on someone for
    childcare?"); it IS one of the 4 comparable battery items used below.
  EXCLUDED: the childcare-ARRANGEMENT / father-care block (NOT wired in here; prose only).
    - Use/arrangement items b19/b20/b21a* (e.g. f2b19 "is child cared for by someone
      other than you on a regular basis?", b20 "# of arrangements", b21a1..a11 who-provides).
    - "partnercare" = father-involvement FREQUENCY items (3pt/4pt scale, -5 'no father').
  Why EXCLUDED, precisely:
    (a) b19 is non-specific: it does NOT distinguish PAID care (daycare / hired) from
        VOLUNTEER kin care. Only unpaid kin care is a village signal; paid care is a
        market transaction -> b19 alone confounds the two and is unusable as-is.
    (b) the frequency items shift scale across waves (3pt/4pt) + carry a known direction
        bug + an informative -5 'no father' structural code (not missing-at-random).
  A CORRECT village-childcare signal would come from the b21a* PROVIDER breakdown,
  restricted to KIN/INFORMAL providers -- grandparents (b21a4 maternal, b21a6 paternal),
  siblings/relatives (b21a3/a5/a7), partners (b21a1/a2/a8/a9) -- and must EXCLUDE the
  paid categories b21a10 (non-relative/family child care) and b21a11 (day care center).
  EMPIRICAL VERDICT (coverage probe on DS0001, N=4898) -- evaluated & EXCLUDED, not deferred:
    * Provider items exist at Y1 (m2b25a*/f2b21a*) and Y3 (m#b8a_*) ONLY -- absent at Y5 & Y9.
      -> fails the cross-wave-repetition gate => cannot be a LAGGED time-varying exposure.
    * Enacted version is selection-gated (asked only if child already in non-parental care):
      Y1 mother 39.5% asked / 20.5% any-kin-yes; Y1 FATHER 1.3% asked / 0.8% (64 fams) -> dead;
      Y3 mother 50.3%/20.7%; Y3 father 21.6%/8.3%.
    * Perceived-trust kin items (m2d3b1/b3 Y1; m#d3a_1/_3 Y3) cover better (47-66% asked) but
      ALSO only Y1 & Y3. Most salvageable use = a BASELINE/Y3 cross-sectional covariate, NOT
      a time-varying village facet. (Active kin co-residence IS captured -- see pipeline
      GRANDPARENT co-residence -- which is structural embeddedness, a different construct.)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from paths import DTA, PROCESSED

# 4 comparable battery items per wave: loan$200, place-to-live, emergency-childcare, co-sign$1000
INSTR = {
    "mother": {2: ["m2g6a","m2g6b","m2g6c","m2g6d"],
               3: ["m3h3","m3h4","m3h5","m3h6"],
               4: ["m4h3","m4h4","m4h5","m4h6"],
               5: ["m5e3","m5e4","m5e5","m5e6"]},
    "father": {2: ["f2g8a","f2g8b","f2g8c","f2g8d"],
               3: ["f3h3","f3h4","f3h5","f3h6"],
               4: ["f4h3","f4h4","f4h5","f4h6"],
               5: ["f5e3","f5e4","f5e5","f5e6"]},
}
REALIZED = {  # gave/loaned money to friends/relatives (kin reciprocity), Y3+
    "mother": {3: "m3l2", 4: "m4l2", 5: "m5j2"},
    "father": {3: "f3l2", 4: "f4l2", 5: "f5j2"},
}

def yn(s):  # FFCWS 1=yes,2=no; negatives = missing
    s = s.where(s >= 0)
    return s.map({1: 1.0, 2: 0.0})

# ---- read only the columns we need (case-mapped) ----
hdr = pd.read_stata(DTA, iterator=True).read(1)
amap = {c.lower(): c for c in hdr.columns}
want = ["idnum"]
for p in INSTR:
    for w in INSTR[p]: want += INSTR[p][w]
    for w in REALIZED[p]: want.append(REALIZED[p][w])
actual = [amap[w] for w in want if w in amap]
miss = [w for w in want if w not in amap]
assert not miss, f"missing cols: {miss}"
raw = pd.read_stata(DTA, columns=actual, convert_categoricals=False)
raw.columns = [c.lower() for c in raw.columns]
raw["idnum"] = raw["idnum"].astype(int)

# ---- build long support table: one row per idnum x parent x wave ----
rows = []
for p in INSTR:
    for w in INSTR[p]:
        items = INSTR[p][w]
        b = raw[items].apply(yn)
        v_instr = b.mean(axis=1).where(b.notna().sum(axis=1) >= 3)  # need >=3 of 4
        rl = REALIZED[p].get(w)
        v_real = yn(raw[rl]) if rl else pd.Series(np.nan, index=raw.index)
        rows.append(pd.DataFrame({"idnum": raw.idnum, "parent": p, "wave": w,
                                  "v_instrumental": v_instr.values, "v_realized": v_real.values}))
support = pd.concat(rows, ignore_index=True)

# ---- merge caseness + covariates + fh ----
L = pd.read_csv(PROCESSED/"ffcws_parental_mh_long.csv")
v = pd.read_csv(PROCESSED/"preexisting_vuln.csv")
fh = (v.melt("idnum", ["fh_mother","fh_father"], var_name="parent", value_name="fh")
        .assign(parent=lambda d: d.parent.str.replace("fh_","")))
base = (L[["idnum","wave","parent","md_case_lib","md_case_con","par_edu","poverty_cat"]]
        .merge(support, on=["idnum","parent","wave"], how="left")
        .merge(fh, on=["idnum","parent"], how="left"))

# ---- assemble lagged-transition panel: exposure at prior wave -> caseness at next ----
PAIRS = [(2,3),(3,4),(4,5)]
out = []
for (wp, wt) in PAIRS:
    prev = base[base.wave==wp].set_index(["idnum","parent"])
    cur  = base[base.wave==wt].set_index(["idnum","parent"])
    d = pd.DataFrame(index=cur.index)
    d["dep_t"]      = cur["md_case_lib"]
    d["dep_t_con"]  = cur["md_case_con"]
    d["dep_prev"]   = prev["md_case_lib"]
    d["v_instrumental"] = prev["v_instrumental"]   # lagged exposure
    d["v_realized"]     = prev["v_realized"]
    d["fh"]         = prev["fh"]
    d["poverty_cat"]= prev["poverty_cat"]
    d["par_edu"]    = prev["par_edu"]
    d["wave_out"]   = wt
    out.append(d.reset_index())
panel = pd.concat(out, ignore_index=True)
panel = panel.dropna(subset=["dep_t","dep_prev"])  # need both ends of the transition
panel.to_csv(PROCESSED/"village_panel_tv.csv", index=False)

# ---- diagnostics ----
print(f"\n=== village_panel_tv.csv written: {len(panel)} rows, {panel.idnum.nunique()} families ===")
print(panel.groupby(["parent","wave_out"]).agg(n=("dep_t","size"),
      dep_rate=("dep_t","mean"), v_instr_cov=("v_instrumental", lambda s: s.notna().mean())).round(3))

print("\n=== WITHIN-PERSON variation in v_instrumental (the identification base) ===")
for p in ["mother","father"]:
    g = panel[(panel.parent==p)].dropna(subset=["v_instrumental"]).groupby("idnum")["v_instrumental"]
    multi = g.count() >= 2
    sd = g.std()[multi]
    print(f"  {p}: {multi.sum()} persons with >=2 support obs | "
          f"{(sd>0).mean()*100:.0f}% of them CHANGE support across waves "
          f"(mean within-person SD {sd.mean():.3f})")

print("\n=== between- vs within-person SD of v_instrumental ===")
for p in ["mother","father"]:
    s = panel[panel.parent==p].dropna(subset=["v_instrumental"])
    pm = s.groupby("idnum")["v_instrumental"].transform("mean")
    print(f"  {p}: between-person SD {pm.std():.3f} | within-person SD {(s.v_instrumental-pm).std():.3f}")
