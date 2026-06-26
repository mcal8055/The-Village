# GRoLTS Checklist — Reporting on Latent Trajectory Studies
van de Schoot, Sijbrandij, Winter, Depaoli & Vermunt (2017), *SEM* 24(3):451–467.

**Scope:** person-centered mixture/latent-trajectory models used *exploratorily* — LGMM (within-class growth variance estimated), LCGA (fixed to zero), GBTM, latent-class growth. **16 items, 21 scorable rows** (subitems a/b/c counted separately); each scored 0 = not reported / 1 = reported. Use as the final compliance pass and as a methods checklist while planning.

| # | Item | What satisfies it |
|---|---|---|
| 1 | Metric of time in the model reported? | State the time coding (age, wave, months-since-event). Decide it a priori from design, never from fit; shape/number of classes is sensitive to follow-up length & spacing. |
| 2 | Mean & variance of time *within* a wave reported? | If data are time-unstructured (people not measured at identical times), report the spread; treating unstructured data as structured biases slopes. |
| 3a | Missing-data mechanism reported? | Name MCAR/MAR/MNAR; models assume MAR (untestable, often reasonable). |
| 3b | Variables related to attrition described? | Compare completers vs. dropouts; identify auxiliary variables. |
| 3c | How missing data were handled? | MI (chained equations / PMM) or FIML, stated explicitly. |
| 4 | Distribution of observed variables reported? | Continuous-normal is often assumed but wrong for binary/ordinal/count/zero-inflated; violations create **spurious classes**. Use the right response model; check measurement invariance over time if using latent indicators. |
| 5 | Software mentioned (and version)? | Name package + version; defaults differ (e.g., Mplus constrains variances across classes by default). |
| 6a | Alternative within-class heterogeneity specs (LCGA vs LGMM) considered/documented? | Show you compared fixing vs. estimating within-class growth variance, or justify excluding one. |
| 6b | Alternative between-class variance–covariance structures considered/documented? | Invariant vs. class-specific residual covariance can change the whole solution; report the choice + rationale. |
| 7 | Alternative trajectory shapes described? | Test linear/quadratic/cubic or nonlinear (logistic, spline, piecewise) against each other. |
| 8 | If covariates used, replicable? | State *where* covariates enter (indicator / growth parameters / class membership) and *how* (one-step vs three-step, BCH/ML). Distinguish conditional vs unconditional. |
| 9 | Number of random starts & final iterations reported? | Likelihood is multimodal → many random starts (≈ **50–100 sets per parameter**, hundreds–thousands for GMM); report starts AND final-stage iterations; flag non-convergence. |
| 10 | Model-comparison tools described statistically? | Name the criteria (BIC, SABIC, AIC, LMR-LRT, BLRT, entropy). Don't rely on one or cherry-pick. |
| 11 | Total number of fitted models reported, including 1-class? | Forward approach from a 1-class solution; fit ≥1–2 models past the apparent optimum; report all. |
| 12 | Cases per class reported for each model? | Absolute n or proportion; tiny classes may be outliers/noise; imbalance distorts estimates. |
| 13 | If classification is the goal, entropy reported? | Relative entropy ∈ [0,1]; ~1 = clean, ~0 = fuzzy (validation median ≈ .85). Use to compare similar-BIC models, **not** to pick k. |
| 14a | Plot of estimated mean trajectories (final solution)? | Show the final class mean trajectories. |
| 14b | Plots of mean trajectories for *each* model? | Show them across the model-building sequence (supplement if needed). |
| 14c | Plot of final-model means + observed individual trajectories, split by class? | Overlay individual data on class means to show how much variability is captured. |
| 15 | Numeric class characteristics (means, SD/SE, n, CI)? | A table, not just plots; note n per parameter (missingness). |
| 16 | Syntax/scripts available? | Appendix, supplement, or permanent repo (OSF) — not solely a personal website. |

**Covariate-uncertainty methods (item 8):** one-step (inflates entropy, changes classes); naive three-step (modal class → biased/attenuated); three-step pseudo-class (MI from posteriors); **bias-adjusted three-step (BCH — robust to distributional violations; ML — not).** Always name which you used. See `covariate_analysis.md`.
