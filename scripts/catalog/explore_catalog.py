#!/usr/bin/env python3
"""
SOURCE-INVENTORY LAYER for the FFCWS feature pipeline.

Treats data/FFMetadata_v20_f.csv (the official FFCWS metadata, 37,322 vars) as the
queryable source catalog. Sweeps every PUBLIC-USE variable (in_FFC_file == "Yes")
that is a candidate for the two engineered constructs, organised by facet, wave and
respondent, and flags:
  - cross-wave repetition  (the gate for a TIME-VARYING measure)
  - coding direction       (value/label pairs; the documented direction-bug hazard)

Outputs (data/catalog/):
  village_candidates.csv   one row per candidate Village (social-support) variable
  famhist_candidates.csv   one row per candidate family MENTAL-HEALTH-history variable
  catalog_inventory.md     human-readable menu: facet x wave coverage + coding

Run:  python3 scripts/catalog/explore_catalog.py

Scoping decisions baked in (confirmed with the researcher):
  * Village = perceived/realized SOCIAL SUPPORT + kin network + grandparent presence.
  * Family history = family MENTAL-HEALTH history ONLY (parental depression / anxiety /
    suicidality) -- NOT general "parents' family background" (education, nativity, etc.).
"""

import re
import numpy as np
import pandas as pd
from paths import METADATA as META, CATALOG as OUT

WAVE_ORDER = {"Baseline": 1, "Year 1": 2, "Year 3": 3,
              "Year 5": 4, "Year 9": 5, "Year 15": 6, "Year 22": 7}
# Current model pipeline (design decision): Baseline through Year 5 ONLY.
# Excludes Year 9 / Year 15 / Year 22.
MODEL_WAVES = ["Baseline", "Year 1", "Year 3", "Year 5"]
WN_MIN, WN_MAX = 1, 4

# ---------------------------------------------------------------------------
# Facet rules. Each facet is matched by (a) optional subtopic filter and
# (b) a regex on the variable LABEL + question text. Kept transparent on purpose:
# this file is the auditable definition of "what counts as Village / family history".
# ---------------------------------------------------------------------------
VILLAGE_FACETS = {
    "instrumental_support": dict(
        subtopics=["social support"],
        label_re=r"count on (someone|anyone).*(loan|\$|place to live|child care|co-?sign)"
                 r"|loan (you )?\$?\d|provide.*place to live|emergency child care|co-?sign",
        note="Perceived instrumental support battery (loan $200/$1000, place to live, "
             "emergency childcare, co-sign $1000/$5000). Repeats every wave.",
    ),
    "realized_transfers": dict(
        subtopics=["private transfers", "social support"],
        label_re=r"(gave|given|receive|received|loan(ed)?|borrow).*(relative|family|friend|kin|parents)"
                 r"|financial (help|support|assistance).*(relative|family|friend|parents)"
                 r"|borrow.*(family|friends)",
        note="Realized kin reciprocity / financial transfers given or received. Y1+.",
    ),
    "trust_network": dict(
        subtopics=["social support", "grandparents", "childcare services and availability"],
        label_re=r"trust.*(look after|watch).*child|who would you trust|count on.*look after",
        note="Structural network breadth: WHO could be trusted to watch the child "
             "(maternal/paternal grandparent, other relative, sibling, friend).",
    ),
    "grandparent_presence": dict(
        subtopics=["grandparents"],
        label_re=r"grandparent|grandmother|grandfather|child see.*parents|see your parents"
                 r"|get along.*parents|parents.*in the hh|in the household",
        note="Grandparent co-residence and child-grandparent contact frequency.",
    ),
}

# All bio-parent mental-health-history items uniquely carry BOTH of these subtopics,
# which isolates them from child anxiety / co-parent mood items. The "she/her" follow-up
# items refer to the bio mother (asked right after the bio-father block in section j).
FAMHIST_SUBTOPIC_AND = ["mental health", "parents' family background"]
FAMHIST_FACETS = {
    "parental_depression": dict(
        label_re=r"(mother|father|she|her).*(depress|down|blue)|(depress|down|blue)",
        note="Bio mother / bio father ever had a 2-week+ depressive period. Y3, repeated Y5.",
    ),
    "parental_anxiety": dict(
        label_re=r"(mother|father|she|her).*(nerv|anx|edgy)|(nerv|anx|edgy)",
        note="Bio parent ever had a month+ of being nervous/anxious. Y3, repeated Y5.",
    ),
    "parental_suicidality": dict(
        label_re=r"suicid",
        note="Bio parent ever attempted suicide. Y3 only.",
    ),
}


def load_meta():
    m = pd.read_csv(META, low_memory=False, encoding="latin-1")
    m = m[m["in_FFC_file"] == "Yes"].copy()
    m["wn"] = m["wave"].map(WAVE_ORDER)
    m["text"] = (m["varlab"].fillna("") + " | " + m["qtext"].fillna(""))
    m["subt"] = m["subtopics"].fillna("").str.lower()
    return m


def coding(row):
    """Compact value:label coding string to eyeball direction (1=yes/2=no vs reversed)."""
    parts = []
    for i in range(1, 5):
        v, l = row.get(f"value{i}"), row.get(f"label{i}")
        if pd.notna(v) and pd.notna(l):
            parts.append(f"{int(v) if float(v).is_integer() else v}={str(l)[:18]}")
    return "; ".join(parts)


def sweep(m, facets, respondents, subtopic_and=None):
    rows = []
    base = m[m["respondent"].isin(respondents) & m["wn"].between(WN_MIN, WN_MAX)]
    if subtopic_and:                       # require ALL of these subtopics present
        for s in subtopic_and:
            base = base[base["subt"].str.contains(re.escape(s), na=False)]
    for facet, rule in facets.items():
        d = base
        if rule.get("subtopics"):          # require ANY of these subtopics
            pat = "|".join(re.escape(s) for s in rule["subtopics"])
            d = d[d["subt"].str.contains(pat, na=False)]
        d = d[d["text"].str.contains(rule["label_re"], case=False, regex=True, na=False)]
        for _, r in d.iterrows():
            rows.append({
                "construct_facet": facet,
                "new_name": r["new_name"], "old_name": r.get("old_name"),
                "wave": r["wave"], "wn": int(r["wn"]), "respondent": r["respondent"],
                "subtopics": r["subtopics"], "scale": r.get("scale"),
                "obs": r.get("obs"), "coding": coding(r),
                "varlab": r["varlab"],
            })
    out = pd.DataFrame(rows).drop_duplicates("new_name").sort_values(
        ["construct_facet", "wn", "respondent", "new_name"])
    return out


def wave_matrix(cand):
    """facet x wave coverage counts (per respondent) -> the time-varying gate."""
    g = (cand.groupby(["construct_facet", "respondent", "wave"]).size()
         .unstack("wave").reindex(columns=MODEL_WAVES).fillna(0).astype(int))
    return g


def write_inventory(village, famhist):
    L = ["# FFCWS construct candidate inventory (source layer)\n",
         "Generated from `data/FFMetadata_v20_f.csv` (public-use vars only). "
         "One row per candidate variable in the CSVs; this file is the menu.\n"]
    for name, cand in [("Village (social support)", village),
                       ("Family mental-health history", famhist)]:
        L.append(f"\n## {name}\n")
        L.append(f"Total candidate variables: **{len(cand)}** "
                 f"across {cand['construct_facet'].nunique()} facets.\n")
        L.append("### Facet × wave coverage (counts; ≥2 waves ⇒ time-varying-capable)\n")
        L.append("```\n" + wave_matrix(cand).to_string() + "\n```\n")
        for facet in cand["construct_facet"].unique():
            d = cand[cand.construct_facet == facet]
            waves = sorted(d.wave.unique(), key=lambda w: WAVE_ORDER[w])
            L.append(f"\n#### `{facet}` — {len(d)} vars; waves: {', '.join(waves)}")
            samp = d.drop_duplicates("varlab").head(8)
            for _, r in samp.iterrows():
                cd = f"  [{r.coding}]" if r.coding else ""
                L.append(f"- `{r.new_name}` ({r.wave}, {r.respondent}) {str(r.varlab)[:78]}{cd}")
    return "\n".join(L)


def main():
    m = load_meta()
    village = sweep(m, VILLAGE_FACETS, ["Mother", "Father"])
    famhist = sweep(m, FAMHIST_FACETS, ["Mother", "Father"],
                    subtopic_and=FAMHIST_SUBTOPIC_AND)
    village.to_csv(f"{OUT}/village_candidates.csv", index=False)
    famhist.to_csv(f"{OUT}/famhist_candidates.csv", index=False)
    with open(f"{OUT}/catalog_inventory.md", "w") as f:
        f.write(write_inventory(village, famhist))

    print(f"Village candidates : {len(village):4d}  -> {OUT}/village_candidates.csv")
    print(village.groupby("construct_facet").size().to_string())
    print(f"\nFamily-history candidates: {len(famhist):4d}  -> {OUT}/famhist_candidates.csv")
    print(famhist.groupby("construct_facet").size().to_string())
    print(f"\nInventory written -> {OUT}/catalog_inventory.md")
    print("\n=== Village facet x wave (mother) ===")
    print(wave_matrix(village).xs("Mother", level="respondent", drop_level=False).to_string())


if __name__ == "__main__":
    main()
