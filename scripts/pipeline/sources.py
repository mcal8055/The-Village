"""
SOURCE LAYER — the single, auditable registry of every raw FFCWS column the
canonical Village / family-history pipeline consumes.

All names were verified present in the public-use .dta header
(data/ICPSR_31622/DS0001/31622-0001-Data.dta) by scripts/catalog/explore_catalog.py
and the validation pass in the build. This module holds NO logic — only the map
from construct -> wave -> source variable. Change a measurement decision HERE and it
propagates to every downstream model, once.

Wave codes (FFCWS prefix numbering): 1=Baseline, 2=Year 1, 3=Year 3, 4=Year 5, 5=Year 9.
Scope: through Year 9 (prefix wave 5) — the LAST wave with CIDI depression caseness for
both parents (Year 15 is not public). Earlier I briefly capped at Y5; restored to Y9 so the
trajectory/persistence outcome keeps all four depression measurements.
"""

from paths import DTA

# Waves at which each construct is measured.
INSTRUMENT_WAVES = [2, 3, 4, 5]       # Y1, Y3, Y5, Y9 (battery not comparable at Baseline)
FH_WAVES = [3, 4]                     # family history asked Y3 with Y5 coverage fill (no Y9)
SUICIDE_WAVES = [3]                   # suicidality asked Y3 only

# ---------------------------------------------------------------------------
# VILLAGE — perceived instrumental support, 6-item battery.
# item-key -> {wave -> {parent -> varname}}.  Names SHIFT across waves (g6/g8 -> h -> e):
# this map is the harmonization that kills the "names shift per wave" hazard.
# All items code 1=yes, 2=no (negatives = missing).
# ---------------------------------------------------------------------------
INSTRUMENTAL = {
    "loan_200":     {2: {"mother": "m2g6a",  "father": "f2g8a"},
                     3: {"mother": "m3h3",   "father": "f3h3"},
                     4: {"mother": "m4h3",   "father": "f4h3"},
                     5: {"mother": "m5e3",   "father": "f5e3"}},
    "loan_1000":    {2: {"mother": "m2g6a1", "father": "f2g8a1"},
                     3: {"mother": "m3h3a",  "father": "f3h3a"},
                     4: {"mother": "m4h3a",  "father": "f4h3a"},
                     5: {"mother": "m5e3a",  "father": "f5e3a"}},
    "place_to_live":{2: {"mother": "m2g6b",  "father": "f2g8b"},
                     3: {"mother": "m3h4",   "father": "f3h4"},
                     4: {"mother": "m4h4",   "father": "f4h4"},
                     5: {"mother": "m5e4",   "father": "f5e4"}},
    "childcare":    {2: {"mother": "m2g6c",  "father": "f2g8c"},
                     3: {"mother": "m3h5",   "father": "f3h5"},
                     4: {"mother": "m4h5",   "father": "f4h5"},
                     5: {"mother": "m5e5",   "father": "f5e5"}},
    "cosign_1000":  {2: {"mother": "m2g6d",  "father": "f2g8d"},
                     3: {"mother": "m3h6",   "father": "f3h6"},
                     4: {"mother": "m4h6",   "father": "f4h6"},
                     5: {"mother": "m5e6",   "father": "f5e6"}},
    "cosign_5000":  {2: {"mother": "m2g6d1", "father": "f2g8d1"},
                     3: {"mother": "m3h6a",  "father": "f3h6a"},
                     4: {"mother": "m4h6a",  "father": "f4h6a"},
                     5: {"mother": "m5e6a",  "father": "f5e6a"}},
}
N_INSTRUMENT_ITEMS = len(INSTRUMENTAL)       # 6
MIN_ITEMS_FOR_SCORE = 4                       # need >=4 of 6 non-missing (cf. old 3-of-4)

# Realized kin transfer — OUTFLOW reciprocity (gave/loaned money TO kin). Y3+ only.
# Retained single item for backward compatibility (the old `v_realized`). 1=yes,2=no.
REALIZED = {3: {"mother": "m3l2", "father": "f3l2"},
            4: {"mother": "m4l2", "father": "f4l2"},
            5: {"mother": "m5j2", "father": "f5j2"}}

# ENACTED kin support — RECEIPT / mobilization (borrowed money from family/friends to
# pay bills, past year). This is the council's "enacted support" facet: a need-driven
# RECEIVED measure expected to correlate POSITIVELY with distress (support mobilization),
# kept as a SEPARATE feature from perceived support — never folded into the index.
# Repeats across all three exposure waves (letter shifts h19i -> i23e -> i23h). 1=yes,2=no.
ENACTED = {2: {"mother": "m2h19i", "father": "f2h17i"},
           3: {"mother": "m3i23e", "father": "f3i23e"},
           4: {"mother": "m4i23h", "father": "f4i23h"},
           5: {"mother": "m5f23g", "father": "f5f23g"}}

# GRANDPARENT co-residence — the structural EMBEDDEDNESS axis of the village (distinct
# from perceived capacity and enacted receipt). Constructed indicators (reliable coding):
# baby's grandfather / grandmother present in the household. Coded 0=no, 1=yes (NOT 1/2).
# v_grandparent = any grandparent in HH. Kept as a SEPARATE sibling facet, never folded in.
# (Re-tested here at the researcher's request: v1 reported a null buffering effect, but v1
# had known issues — the canonical layer lets the updated models confirm or revise that.)
GRANDPARENT = {
    2: {"mother": ("cm2gdad", "cm2gmom"), "father": ("cf2gdad", "cf2gmom")},
    3: {"mother": ("cm3gdad", "cm3gmom"), "father": ("cf3gdad", "cf3gmom")},
    4: {"mother": ("cm4gdad", "cm4gmom"), "father": ("cf4gdad", "cf4gmom")},
    5: {"mother": ("cm5gdad", "cm5gmom"), "father": ("cf5gdad", "cf5gmom")},
}

# RELIGIOUS PARTICIPATION — attendance at religious services. A separate, ablatable village
# facet (community engagement), NOT folded into perceived support. Behavioral participation
# only (attendance), deliberately distinct from internal religiosity/salience or affiliation.
# HAZARD: the raw item is a REVERSE-coded ordinal whose category count + definitions SHIFT
# across waves, and Y1 carries a parallel questionnaire-version coding (201-205 = version B
# of the same scale). An ordinal pool is therefore untrustworthy; we harmonize to a binary
# "attends weekly or more" using explicit per-wave value sets (the only correct way to fold
# Y1 version-B's 201 in with version-A's 1,2). 1 = weekly+, 0 = less often, NaN = missing.
RELIGION = {2: {"mother": "m2g4c", "father": "f2g6c"},
            3: {"mother": "m3r10", "father": "f3r10"},
            4: {"mother": "m4r2",  "father": "f4r2"},
            5: {"mother": "m5h2",  "father": "f5h2"}}
RELIGION_WEEKLY = {2: {1, 2, 201}, 3: {1, 2, 3}, 4: {1, 2, 3}, 5: {1, 2, 3}}

# ---------------------------------------------------------------------------
# FAMILY MENTAL-HEALTH HISTORY — graded, multi-symptom.
# symptom -> respondent-parent -> wave -> bio-parent -> (base, treated, hospitalized)
# base/treated/hosp code 1=yes,2=no. treated/hosp are None where not asked (suicide).
# NOTE the irregular naming: father's bio-father depression follow-ups are j53a/j53b
# (NOT j52a/j52b). This map is the authority; do not infer names by pattern.
# ---------------------------------------------------------------------------
FAMILY_HISTORY = {
    "depression": {
        "mother": {3: {"biofather": ("m3j45", "m3j45a", "m3j45b"),
                       "biomother": ("m3j50", "m3j50a", "m3j50b")},
                   4: {"biofather": ("m4j26", "m4j26a", "m4j26b"),
                       "biomother": ("m4j31", "m4j31a", "m4j31b")}},
        "father": {3: {"biofather": ("f3j52", "f3j53a", "f3j53b"),
                       "biomother": ("f3j58", "f3j58a", "f3j58b")},
                   4: {"biofather": ("f4j26", "f4j26a", "f4j26b"),
                       "biomother": ("f4j31", "f4j31a", "f4j31b")}},
    },
    "anxiety": {
        "mother": {3: {"biofather": ("m3j46", "m3j46a", "m3j46b"),
                       "biomother": ("m3j51", "m3j51a", "m3j51b")},
                   4: {"biofather": ("m4j27", "m4j27a", "m4j27b"),
                       "biomother": ("m4j32", "m4j32a", "m4j32b")}},
        "father": {3: {"biofather": ("f3j54", "f3j54a", "f3j54b"),
                       "biomother": ("f3j59", "f3j59a", "f3j59b")},
                   4: {"biofather": ("f4j27", "f4j27a", "f4j27b"),
                       "biomother": ("f4j32", "f4j32a", "f4j32b")}},
    },
    "suicide": {
        "mother": {3: {"biofather": ("m3j49", None, None),
                       "biomother": ("m3j54", None, None)}},
        "father": {3: {"biofather": ("f3j57", None, None),
                       "biomother": ("f3j62", None, None)}},
    },
}

# Father baseline CES-D (12 short items, 'days in past week'); summed for continuity
# with the existing cesd_father_baseline column. No mother equivalent in the public file.
CESD_FATHER_BASELINE = [f"f1g9{c}" for c in "abcdefghijkl"]


def all_source_columns():
    """Flat set of every raw column this pipeline reads — for the freshness/contract check."""
    cols = {"idnum"}
    for item in INSTRUMENTAL.values():
        for wave in item.values():
            cols.update(wave.values())
    for wave in REALIZED.values():
        cols.update(wave.values())
    for wave in ENACTED.values():
        cols.update(wave.values())
    for wave in GRANDPARENT.values():
        for pair in wave.values():
            cols.update(pair)
    for wave in RELIGION.values():
        cols.update(wave.values())
    for sym in FAMILY_HISTORY.values():
        for par in sym.values():
            for wave in par.values():
                for bp in wave.values():
                    cols.update(n for n in bp if n)
    cols.update(CESD_FATHER_BASELINE)
    return cols
