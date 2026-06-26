"""
STAGING LAYER — thin wrappers over the raw .dta. One concern only: read each source
column once, recode FFCWS missing codes, harmonize the wave-shifting names into stable
long tables. NO business logic / scoring here (that lives in intermediate.py).

Every staging function returns a tidy long frame keyed by (idnum, parent[, wave, ...]).
"""
import numpy as np
import pandas as pd

from . import sources as S


def _yn(s):
    """FFCWS 1=yes, 2=no; all negatives are missing codes -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s >= 0)
    return s.map({1: 1.0, 2: 0.0})


def _bin01(s):
    """Already-0/1 constructed indicator; negatives -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0)


def read_raw():
    """Read ONLY the registered source columns from the .dta, once. ICPSR stores names
    upper-case; map case-insensitively. Fails loudly if any registered column is absent."""
    hdr = pd.read_stata(S.DTA, iterator=True).read(1)
    lower2actual = {c.lower(): c for c in hdr.columns}
    want = S.all_source_columns()
    missing = sorted(w for w in want if w not in lower2actual)
    if missing:
        raise KeyError(f"Source columns absent from .dta (schema drift?): {missing}")
    actual = [lower2actual[w] for w in want]
    raw = pd.read_stata(S.DTA, columns=actual, convert_categoricals=False)
    raw.columns = [c.lower() for c in raw.columns]
    raw["idnum"] = raw["idnum"].astype(int)
    return raw


def stg_instrumental(raw):
    """Long: one row per idnum x parent x wave x battery-item, value in {0,1,NaN}."""
    rows = []
    for item, waves in S.INSTRUMENTAL.items():
        for wave, parents in waves.items():
            for parent, var in parents.items():
                rows.append(pd.DataFrame({
                    "idnum": raw["idnum"], "parent": parent, "wave": wave,
                    "item": item, "value": _yn(raw[var]).values,
                }))
    return pd.concat(rows, ignore_index=True)


def stg_realized(raw):
    """Long: idnum x parent x wave -> v_realized (single retained item; Y3+ only)."""
    rows = []
    for wave, parents in S.REALIZED.items():
        for parent, var in parents.items():
            rows.append(pd.DataFrame({
                "idnum": raw["idnum"], "parent": parent, "wave": wave,
                "v_realized": _yn(raw[var]).values,
            }))
    return pd.concat(rows, ignore_index=True)


def stg_enacted(raw):
    """Long: idnum x parent x wave -> v_enacted (borrowed from kin to pay bills; receipt)."""
    rows = []
    for wave, parents in S.ENACTED.items():
        for parent, var in parents.items():
            rows.append(pd.DataFrame({
                "idnum": raw["idnum"], "parent": parent, "wave": wave,
                "v_enacted": _yn(raw[var]).values,
            }))
    return pd.concat(rows, ignore_index=True)


def stg_grandparent(raw):
    """Long: idnum x parent x wave -> v_grandparent (any grandparent in HH, 0/1).
    NaN only when both gdad and gmom are missing (they share a missingness pattern)."""
    rows = []
    for wave, parents in S.GRANDPARENT.items():
        for parent, (gdad, gmom) in parents.items():
            gd, gm = _bin01(raw[gdad]), _bin01(raw[gmom])
            # any-grandparent: max ignoring NaN; NaN only if both missing
            v = pd.concat([gd, gm], axis=1).max(axis=1)
            v = v.where(gd.notna() | gm.notna())
            rows.append(pd.DataFrame({
                "idnum": raw["idnum"], "parent": parent, "wave": wave,
                "v_grandparent": v.values,
            }))
    return pd.concat(rows, ignore_index=True)


def stg_religion(raw):
    """Long: idnum x parent x wave -> v_religion (1 = attends religious services weekly+,
    0 = less often, NaN = missing). Resolves the per-wave scale shift AND Y1's parallel
    version-B coding via the explicit weekly-value sets in sources.RELIGION_WEEKLY."""
    rows = []
    for wave, parents in S.RELIGION.items():
        weekly = list(S.RELIGION_WEEKLY[wave])
        for parent, var in parents.items():
            x = pd.to_numeric(raw[var], errors="coerce")
            v = x.isin(weekly).astype(float).where(x >= 0)   # negatives -> NaN
            rows.append(pd.DataFrame({
                "idnum": raw["idnum"], "parent": parent, "wave": wave,
                "v_religion": v.values,
            }))
    return pd.concat(rows, ignore_index=True)


def stg_family_history(raw):
    """Long: idnum x parent x symptom x bio_parent x wave, with an ordinal severity level:
    0=no history, 1=history, 2=treated, 3=hospitalized; NaN if the item was not answered."""
    rows = []
    for symptom, resp in S.FAMILY_HISTORY.items():
        for parent, waves in resp.items():
            for wave, bios in waves.items():
                for bio_parent, (base, treated, hosp) in bios.items():
                    b = _yn(raw[base])
                    t = _yn(raw[treated]) if treated else pd.Series(np.nan, index=raw.index)
                    h = _yn(raw[hosp]) if hosp else pd.Series(np.nan, index=raw.index)
                    # severity ladder: start at base (0/1), escalate on treated/hosp.
                    level = b.copy()
                    level = level.mask(t == 1, 2)
                    level = level.mask(h == 1, 3)
                    rows.append(pd.DataFrame({
                        "idnum": raw["idnum"], "parent": parent, "symptom": symptom,
                        "bio_parent": bio_parent, "wave": wave,
                        "base": b.values, "level": level.values,
                    }))
    return pd.concat(rows, ignore_index=True)


def stg_cesd_father_baseline(raw):
    """Father baseline CES-D sum (12 items; NaN if any item missing — matches prior build)."""
    ces = raw[S.CESD_FATHER_BASELINE].apply(lambda c: pd.to_numeric(c, errors="coerce").where(lambda x: x >= 0))
    return pd.DataFrame({"idnum": raw["idnum"],
                         "cesd_father_baseline": ces.sum(axis=1, min_count=len(S.CESD_FATHER_BASELINE))})
