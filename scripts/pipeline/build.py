#!/usr/bin/env python3
"""
Canonical FFCWS feature pipeline — DAG runner.

  raw .dta
    -> dataset   (parental-MH caseness panel -> ffcws_parental_mh_long.csv)
    -> staging   (recode-once, harmonize wave-shifting names)
    -> intermediate (Village instrumental score; graded family-history)
    -> marts     (preexisting_vuln.csv, village_panel_tv.csv)

Run from the repo root:  python3 -m scripts.pipeline.build
"""
from . import build_ffcws_dataset as caseness
from . import staging as stg
from . import intermediate as itm
from . import marts as mt


def run():
    print("[dataset]  building parental-MH caseness panel from raw .dta ...")
    caseness.build()
    print("[staging]  reading raw .dta (registered source columns only) ...")
    raw = stg.read_raw()
    s_instr = stg.stg_instrumental(raw)
    s_real = stg.stg_realized(raw)
    s_enac = stg.stg_enacted(raw)
    s_gp = stg.stg_grandparent(raw)
    s_rel = stg.stg_religion(raw)
    s_fh = stg.stg_family_history(raw)
    s_cesd = stg.stg_cesd_father_baseline(raw)
    print(f"           {len(raw):,} families | instrumental rows {len(s_instr):,} "
          f"| family-history rows {len(s_fh):,}")

    print("[interm.]  building canonical constructs ...")
    village = itm.int_village_instrumental(s_instr)
    fh = itm.int_family_history(s_fh)

    print("[marts]    materialising drop-in CSVs ...")
    dim_fh = mt.dim_family_vulnerability(fh, s_cesd)
    panel = mt.fct_village_panel_tv(village, s_real, s_enac, s_gp, s_rel, dim_fh)
    mt.fct_village_wave(village, s_real, s_enac, s_gp, s_rel)
    person = mt.obt_model_person(village, s_real, s_enac, s_gp, s_rel, dim_fh)

    # ---- reconciliation diagnostics ----
    print("\n=== dim_family_vulnerability (preexisting_vuln.csv) ===")
    print(f"  families: {len(dim_fh):,}")
    for p in ("mother", "father"):
        print(f"  {p}: fh_any={dim_fh[f'fh_{p}'].mean():.3f} (n={dim_fh[f'fh_{p}'].notna().sum()}) "
              f"| burden mean={dim_fh[f'fh_{p}_burden'].mean():.2f} "
              f"| severity mean={dim_fh[f'fh_{p}_severity'].mean():.2f} "
              f"| suicide={dim_fh[f'fh_{p}_suicide'].mean():.3f}")
    print(f"  cesd_father_baseline: mean={dim_fh['cesd_father_baseline'].mean():.2f} "
          f"(n={dim_fh['cesd_father_baseline'].notna().sum()})")

    print("\n=== fct_village_panel_tv (village_panel_tv.csv) ===")
    print(f"  rows: {len(panel):,} | families: {panel.idnum.nunique():,}")
    print(panel.groupby(["parent", "wave_out"]).agg(
        n=("dep_t", "size"), dep_rate=("dep_t", "mean"),
        v_perceived_cov=("v_perceived", lambda s: s.notna().mean()),
        v_enacted_cov=("v_enacted", lambda s: s.notna().mean())).round(3).to_string())

    # bracketing check: do perceived and enacted point opposite ways vs dep_prev?
    print("\n=== perceived vs enacted: raw association with PRIOR depression (sign check) ===")
    for p in ("mother", "father"):
        d = panel[panel.parent == p]
        rp = d[["v_perceived", "dep_prev"]].corr().iloc[0, 1]
        re_ = d[["v_enacted", "dep_prev"]].corr().iloc[0, 1]
        recv = d["v_enacted"].mean()
        disc = d["enacted_discordant"].mean()
        print(f"  {p}: corr(perceived,dep_prev)={rp:+.3f}  corr(enacted,dep_prev)={re_:+.3f}"
              f"  | enacted-receipt rate={recv:.3f}  discordant-share-of-receivers={disc:.3f}")

    print("\n=== grandparent embeddedness (re-test of v1's null) ===")
    for p in ("mother", "father"):
        d = panel[panel.parent == p].dropna(subset=["v_grandparent"])
        prev_rate = d["v_grandparent"].mean()
        rg = d[["v_grandparent", "dep_t"]].corr().iloc[0, 1]
        # crude lagged signal: next-wave depression rate by prior grandparent-in-HH
        by = d.groupby("v_grandparent")["dep_t"].mean()
        print(f"  {p}: grandparent-in-HH rate={prev_rate:.3f} (n={len(d)}) "
              f"| corr(gp_prev,dep_t)={rg:+.3f} "
              f"| dep_t | gp=0:{by.get(0.0, float('nan')):.3f}  gp=1:{by.get(1.0, float('nan')):.3f}")

    print("\n=== religion participation (new ablation facet: weekly+ attendance) ===")
    for p in ("mother", "father"):
        d = panel[panel.parent == p].dropna(subset=["v_religion"])
        by = d.groupby("v_religion")["dep_t"].mean()
        print(f"  {p}: weekly+ rate={d['v_religion'].mean():.3f} (n={len(d)}) "
              f"| corr(relig_prev,dep_t)={d[['v_religion','dep_t']].corr().iloc[0,1]:+.3f} "
              f"| dep_t | relig=0:{by.get(0.0, float('nan')):.3f}  relig=1:{by.get(1.0, float('nan')):.3f}")

    print("\n=== obt_model_person (predictive mart) ===")
    print(f"  rows: {len(person):,} | families: {person.idnum.nunique():,}")
    print(person.groupby("parent").agg(
        n=("persistent", "size"), persistent=("persistent", "mean"),
        support_mean=("support_mean", "mean"), enacted_mean=("enacted_mean", "mean"),
        grandparent_mean=("grandparent_mean", "mean"), fh=("fh", "mean")).round(3).to_string())
    return dim_fh, panel, person


if __name__ == "__main__":
    run()
