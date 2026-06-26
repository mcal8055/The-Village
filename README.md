# The Village

**Cause or Signature?** A causal analysis of social support ("the Village") and parental depression in the **Fragile Families and Child Wellbeing Study (FFCWS)**.

## The question

When a parent has people they can lean on — someone to lend $200, offer a place to stay, watch the kids in an emergency — are they *protected* from depression? Or do low support and depression simply travel together because both flow from a shared, stable vulnerability?


## Approach

The analysis separates a *descriptive* question (what is the structure of parental depression over time?) from a *causal* one (does support move it?), and answers the causal question with a **four-rung ladder** that climbs from the weakest claim to the strongest — each rung closing the loophole the one beneath it leaves open.

| Rung | Test | What it removes / answers |
|:---:|---|---|
| 1 | Within-person (Mundlak + fixed-effects logit) | *all time-invariant* confounding — each parent is their own control |
| 2 | E-value sensitivity | how strong an unmeasured confounder would have to be to erase the surviving effect |
| 3 | Cross-lagged reverse test | reverse causation (depression → lower future support) |
| 4 | Terminal new-onset subgroup | tests the thesis directly in its cleanest pro-thesis slice |

Upstream of the ladder: latent **trajectory clustering** (group-based trajectory modeling) identifies a persistent vs. rarely-depressed subtype, and a Stage-2 model asks what predicts persistence. Crucially, the cluster labels never enter the causal ladder, so the causal verdict is independent of the (fuzzy) clustering.

## Headline finding

Parental depression is **trait-like / vulnerability-driven**, not a perinatal, temporal spike. **Family-history vulnerability dominates SES** in predicting persistence. Social support is **protective between persons but null within persons** — across all four facets (perceived, enacted, grandparent co-residence, religious participation), under both estimators, for mothers and fathers, with a symmetric null in the reverse direction.

Read together, that is the signature of a **stable common cause**, not a causal lever: the Village is most plausibly a *marker* of an underlying disposition, not something that *moves* parental depression within a person. (Observational; the within-person design removes time-invariant confounding only — this is "no within-person effect after removing stable confounding," not a proven zero.)

## Repository layout

```
paths.py                          Canonical filesystem anchors
conftest.py                       Puts repo root on sys.path (imports + pytest)
scripts/pipeline/                 dbt-style data pipeline (the single source of truth)
  sources.py                      raw FFCWS column registry (source layer)
  staging.py / intermediate.py    harmonization + engineered constructs
  marts.py                        analysis-ready marts (caseness panel, vulnerability,
                                  village facets, lagged-transition panel)
  build_ffcws_dataset.py          builds the upstream CIDI caseness panel from the raw .dta
  build.py                        `python3 -m scripts.pipeline.build`
  tests/test_contract.py          dbt-style data tests (unique / not-null / ranges / reconciliation)
FFCWS_Village_Analysis.Rmd        the consolidated causal analysis (clustering + the ladder)
skills/
  longitudinal-cluster-analysis/  the GBTM/GRoLTS trajectory-clustering method behind Stage 1
```

## Data (not included)

This repository is **code only**. FFCWS data is governed by a data-use agreement and is **not redistributed here** — neither the raw file nor any derived row-level output.

The pipeline consumes the FFCWS **public-use** file (ICPSR study **31622**). To reproduce, obtain it yourself and place it at:

```
data/ICPSR_31622/DS0001/31622-0001-Data.dta
data/FFMetadata_v20_f.csv
```

Everything under `data/` is gitignored. ICPSR: <https://www.icpsr.umich.edu/web/ICPSR/studies/31622>; study site: <https://ffcws.princeton.edu/>.

## Reproducing

```bash
# 1. Build the canonical marts from the raw .dta (writes to data/processed/)
python3 -m scripts.pipeline.build

# 2. Validate the marts (dbt-style data contract)
pytest scripts/pipeline/tests -q

# 3. Render the causal analysis
Rscript -e 'rmarkdown::render("FFCWS_Village_Analysis.Rmd")'
```

### Dependencies

- **Python**: `pandas`, `numpy`, `stepmix`, `statsmodels`, `scikit-learn`, `interpret`, `pytest`
- **R**: `rmarkdown`, `knitr`, `haven`, `dplyr`, `tidyr`, `ggplot2`, `flexmix`, `geepack`, `survival`

## Status

This is the **causal** analysis. A companion **predictive** re-analysis (supervised ML cross-check of the same drivers) is being prepared and will be added once it has had the same audit and review pass.
