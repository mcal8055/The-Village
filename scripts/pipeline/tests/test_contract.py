"""
Data tests for the canonical marts — the dbt-style `unique / not_null / accepted_values
/ relationships / range` checks, plus N reconciliation and the prevalence/coverage
expectations from contract.yml.

Run:  pytest scripts/pipeline/tests -q
The marts must be built first:  python3 -m scripts.pipeline.build
"""
import os
import pandas as pd
import pytest
from paths import PROCESSED as OUT

VULN = f"{OUT}/preexisting_vuln.csv"
PANEL = f"{OUT}/village_panel_tv.csv"
PERSON = f"{OUT}/obt_model_person.csv"
LONG = f"{OUT}/ffcws_parental_mh_long.csv"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VULN) and os.path.exists(PANEL)),
    reason="marts not built — run `python3 -m scripts.pipeline.build` first")


@pytest.fixture(scope="module")
def vuln():
    return pd.read_csv(VULN)


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PANEL)


@pytest.fixture(scope="module")
def person():
    return pd.read_csv(PERSON)


# ---- dim_family_vulnerability ----
def test_vuln_pk_unique_not_null(vuln):
    assert vuln["idnum"].is_unique
    assert vuln["idnum"].notna().all()

def test_vuln_family_count(vuln):
    assert len(vuln) == 4898                      # reconcile to source N

def test_fh_binary_accepted_values(vuln):
    for c in ["fh_mother", "fh_father", "fh_mother_suicide", "fh_father_suicide"]:
        assert set(vuln[c].dropna().unique()) <= {0.0, 1.0}, c

def test_fh_burden_range(vuln):
    for c in ["fh_mother_burden", "fh_father_burden"]:
        assert vuln[c].dropna().between(0, 6).all(), c

def test_fh_severity_range(vuln):
    for c in ["fh_mother_severity", "fh_father_severity"]:
        assert vuln[c].dropna().between(0, 3).all(), c

def test_cesd_non_negative(vuln):
    assert (vuln["cesd_father_baseline"].dropna() >= 0).all()

def test_fh_prevalence_expectations(vuln):
    assert 0.30 <= vuln["fh_mother"].mean() <= 0.37
    assert 0.26 <= vuln["fh_father"].mean() <= 0.33

def test_burden_consistent_with_any(vuln):
    # if either bio-parent was ever depressed (fh_any=1), burden must be >= 1
    sub = vuln[vuln["fh_mother"] == 1]
    assert (sub["fh_mother_burden"] >= 1).all()


# ---- fct_village_panel_tv ----
def test_panel_pk_unique(panel):
    assert not panel.duplicated(subset=["idnum", "parent", "wave_out"]).any()

def test_panel_outcomes_not_null_binary(panel):
    for c in ["dep_t", "dep_prev"]:
        assert panel[c].notna().all()
        assert set(panel[c].unique()) <= {0.0, 1.0}

def test_panel_v_instrumental_range(panel):
    v = panel["v_instrumental"].dropna()
    assert v.between(0, 1).all()

def test_perceived_is_instrumental_alias(panel):
    # v_perceived must be an exact alias of v_instrumental (same NaNs, same values)
    a, b = panel["v_perceived"], panel["v_instrumental"]
    assert a.isna().equals(b.isna())
    assert (a.dropna() == b.dropna()).all()

def test_enacted_binary(panel):
    assert set(panel["v_enacted"].dropna().unique()) <= {0.0, 1.0}

def test_grandparent_binary(panel):
    assert set(panel["v_grandparent"].dropna().unique()) <= {0.0, 1.0}

def test_religion_binary(panel):
    assert set(panel["v_religion"].dropna().unique()) <= {0.0, 1.0}

def test_village_facets_are_separate_columns(panel):
    # architecture invariant: the axes coexist as distinct columns,
    # and there is NO single folded composite 'v_village' score.
    for c in ["v_perceived", "v_enacted", "v_grandparent", "v_religion"]:
        assert c in panel.columns
    assert "v_village" not in panel.columns

def test_discordant_only_defined_for_receivers(panel):
    # enacted_discordant is non-null iff enacted receipt occurred (v_enacted == 1)
    assert set(panel["enacted_discordant"].dropna().unique()) <= {0.0, 1.0}
    assert (panel.loc[panel["enacted_discordant"].notna(), "v_enacted"] == 1).all()

def test_perceived_enacted_opposite_sign(panel):
    # core empirical claim: enacted correlates MORE positively with prior
    # depression than perceived does -> they must not be folded into one index.
    rp = panel[["v_perceived", "dep_prev"]].corr().iloc[0, 1]
    re_ = panel[["v_enacted", "dep_prev"]].corr().iloc[0, 1]
    assert re_ > rp

def test_panel_v_instrumental_coverage(panel):
    assert panel["v_instrumental"].notna().mean() >= 0.90

def test_panel_wave_out_accepted(panel):
    assert set(panel["wave_out"].unique()) <= {3, 4, 5}    # Y3, Y5, Y9 outcomes

def test_panel_parent_accepted(panel):
    assert set(panel["parent"].unique()) == {"mother", "father"}

# ---- obt_model_person (predictive mart) ----
def test_person_pk_unique(person):
    assert not person.duplicated(subset=["idnum", "parent"]).any()

def test_person_has_required_feature_columns(person):
    required = ["persistent", "fh", "support_y1", "support_mean", "realized_mean",
                "enacted_mean", "grandparent_mean", "fh_burden", "fh_severity", "fh_suicide",
                "poverty_cat", "par_edu", "hhinc", "ethrace", "gad_y1", "cesd_father_baseline"]
    missing = [c for c in required if c not in person.columns]
    assert not missing, f"person mart missing: {missing}"

def test_person_target_binary(person):
    assert set(person["persistent"].unique()) <= {0, 1}

def test_person_min_two_waves(person):
    assert (person["n_waves"] >= 2).all()

def test_person_new_facets_in_range(person):
    for c in ["support_mean", "enacted_mean", "grandparent_mean"]:
        assert person[c].dropna().between(0, 1).all(), c

def test_person_father_only_cesd(person):
    # mothers have no baseline CES-D; fathers do
    assert person.loc[person.parent == "mother", "cesd_father_baseline"].isna().all()
    assert person.loc[person.parent == "father", "cesd_father_baseline"].notna().any()


@pytest.mark.skipif(not os.path.exists(LONG), reason="upstream caseness long not present")
def test_panel_keys_exist_in_caseness(panel):
    # relationships test: every (idnum,parent) in the panel exists upstream
    long = pd.read_csv(LONG)[["idnum", "parent"]].drop_duplicates()
    merged = panel[["idnum", "parent"]].drop_duplicates().merge(
        long, on=["idnum", "parent"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
