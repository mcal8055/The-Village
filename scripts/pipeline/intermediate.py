"""
INTERMEDIATE LAYER — the two engineered constructs, each defined exactly ONCE.

  int_village_instrumental : harmonized 6-item perceived-support score per wave.
  int_family_history       : graded, multi-symptom family mental-health-history per parent.

These are the canonical definitions. Every downstream model reads these, never the raw.
"""
import numpy as np
import pandas as pd

from . import sources as S


def int_village_instrumental(stg_instr):
    """Per idnum x parent x wave: mean of available battery items (0..1), requiring
    >= MIN_ITEMS_FOR_SCORE of the 6 items non-missing (else NaN). Carries the item count."""
    g = stg_instr.groupby(["idnum", "parent", "wave"])["value"]
    out = g.agg(n_items=lambda s: s.notna().sum(),
                v_instrumental=lambda s: s.mean()).reset_index()
    out.loc[out["n_items"] < S.MIN_ITEMS_FOR_SCORE, "v_instrumental"] = np.nan
    return out


def _union_over_waves(stg_fh):
    """Collapse Y3/Y5 to one row per idnum x parent x symptom x bio_parent:
    base = endorsed at EITHER wave (NaN only if never observed); level = max severity seen.
    This is the documented Y3-with-Y5-coverage-fill, applied per cell."""
    def agg(s):
        v = s.dropna()
        return np.nan if v.empty else v.max()
    return (stg_fh.groupby(["idnum", "parent", "symptom", "bio_parent"])
            .agg(base=("base", agg), level=("level", agg)).reset_index())


def int_family_history(stg_fh):
    """Per idnum x parent: the graded family mental-health-history construct.
      fh_any      : either bio-parent ever DEPRESSED (backward-compatible with prior `fh`)
      fh_burden   : count of endorsed (symptom x bio-parent) cells, 0..6
      fh_severity : max treatment-intensity level across depression+anxiety cells, 0..3
      fh_suicide  : either bio-parent ever attempted suicide
    A parent with NO family-history items answered gets NaN across the board (no module)."""
    cell = _union_over_waves(stg_fh)

    def per_parent(d):
        observed = d["base"].notna().any()
        dep = d[d.symptom == "depression"]["base"]
        sev = d[d.symptom.isin(["depression", "anxiety"])]["level"]
        suic = d[d.symptom == "suicide"]["base"]
        return pd.Series({
            "fh_any": np.nan if dep.dropna().empty else float(dep.max() == 1),
            "fh_burden": np.nan if not observed else float((d["base"] == 1).sum()),
            "fh_severity": np.nan if sev.dropna().empty else float(sev.max()),
            "fh_suicide": np.nan if suic.dropna().empty else float(suic.max() == 1),
        })

    return cell.groupby(["idnum", "parent"]).apply(per_parent, include_groups=False).reset_index()
