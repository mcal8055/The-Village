---
name: longitudinal-cluster-analysis
description: >-
  Group individuals into subtypes by their TRAJECTORIES over repeated measures (longitudinal / panel / time-series-of-people data). Use this whenever the user wants to find latent trajectory classes or developmental subtypes over time, run a growth mixture model (GMM), latent class growth analysis (LCGA) or group-based trajectory model (GBTM), cluster repeated-measures or panel data, do longitudinal k-means (KmL) or sequence analysis, decide how many trajectory classes there are, profile or predict class membership, validate trajectory clusters, or report a latent-trajectory study (GRoLTS). Trigger even when the user only says things like "are there distinct groups of patients over time", "subtypes of responders", "clusters of trajectories", "do people follow different paths", or names a panel/cohort with several waves and asks how it splits into groups. Prefer this skill over generic clustering whenever a time/wave dimension is involved.
---

# Longitudinal Cluster Analysis

Clustering longitudinal data means grouping *units* (people, patients, firms…) by the **shape of their trajectory** across repeated measurements — not by their values at one time point. The goal is to turn a continuous spectrum of heterogeneity into a small number of interpretable trajectory subtypes (e.g., *resilient / recovering / chronic*).

This skill encodes a defensible end-to-end workflow (after Den Teuling et al., 2026) and the GRoLTS reporting standard (van de Schoot et al., 2017). Follow the 7-step workflow; reach for the reference files for depth; use the bundled script for the Python-feasible methods.

## Read-this-first principles

These five ideas prevent the most common ways longitudinal cluster analyses mislead. Internalize them before fitting anything.

1. **Clustering here is usually *indirect* (approximation), not *direct* (discovery of real groups).** Most populations vary on a continuum with no truly distinct classes. Treating clusters as a convenient summary of heterogeneity is honest and robust; claiming you've found *real, distinct* subtypes is a much stronger claim that requires large between-cluster separation and confirmatory validation. Default to the indirect interpretation and say so. Substantial overlap can still be useful indirectly but discredits a direct claim.

2. **Fit the one-cluster model first.** If a single trajectory (with random effects) already fits well, extra clusters add little. The 1-class model also reveals the approximate trajectory shape you'll need later, and it is the mandatory baseline for class enumeration (GRoLTS item 11).

3. **Validate, because these models overfit.** Parameters scale *linearly* with the number of clusters, so on modest samples a k-class solution can be a mirage that won't replicate. This is also where the **circular-analysis ("double-dipping") trap** lives: if you choose k by exploring the data and then "confirm" associations on the *same* data, your inference is invalid. Guard against both with a **holdout/validation sample**, **k-fold CV**, and/or **bootstrap stability** of the solution. Rarely done in practice — do it anyway.

4. **When you relate clusters to covariates, correct for classification uncertainty.** Membership is *probabilistic*. Assigning each unit to its modal class and then running ANOVA / multinomial regression on those hard labels ignores that uncertainty and **attenuates** covariate effects. Use a bias-adjusted three-step approach (BCH or ML) — see `references/covariate_analysis.md`.

5. **Report to GRoLTS.** Latent-trajectory studies are notoriously under-reported (the GRoLTS validation sample met ~9 of 21 items on average). Plan to satisfy the 21-point checklist from the start; it doubles as a methods checklist. See `references/grolts_checklist.md`.

## The workflow

Work through these seven steps. Steps 1–4 are scoping you do *before* fitting; step 5 is the iterative modeling core; steps 6–7 are inference and interpretation.

### 1. Analyze the model variables
Establish the **response type and distribution** — continuous-normal, but also binary, ordinal, count, zero-inflated, censored/truncated. This determines which methods are even valid: assuming continuous-normal when the response is discrete or skewed can manufacture **spurious extra classes** (Bauer & Curran). Also inspect **covariate distributions and outliers**, which can distort distance- and feature-based methods. Profile the data (the `programmatic-eda` skill is a fine companion here).

### 2. Investigate the missing-data mechanism
Report whether data are MCAR / MAR / MNAR and what relates to attrition. Mixture models assume **MAR** (untestable but often reasonable for repeated measures), and a strength of clustering is that missingness *driven by the trajectory itself* is partly absorbed by the classes. For MNAR, consider pattern-mixture / selection models. Compare completers vs. dropouts and note auxiliary variables. Distinguish **time-structured** (everyone measured at the same occasions) from **time-unstructured** data — ignoring varying measurement times biases slopes.

### 3. Model the data as a single cluster
Fit the 1-class model (a plain growth/mixed model). Judge its fit and read off the dominant trajectory shape. A good 1-class fit is a signal to be skeptical of multi-class stories.

### 4. Provide a rationale for clustering, and choose direct vs. indirect
State *why* you expect subtypes (theory, prior studies) and commit to the **indirect** (approximation) or **direct** (confirmatory) interpretation — it governs how you'll judge and report the result (principle 1).

### 5. Identify the best model
The intricate core. Start with an **unconditional model** (time only, no covariates) so the classes reflect trajectory shape, not covariate structure. Iterate over these sub-steps:

- **5a. Choose the method.** Weigh trajectory-shape flexibility, whether you need within-cluster heterogeneity, sample-size needs, and compute cost. See the selection guide below and `references/methods_taxonomy.md`.
- **5b. Choose estimation & run many random starts.** EM/ML mixtures are multimodal and routinely land on **local optima**. Rerun with many random starting sets (GRoLTS guidance: *50–100 sets per parameter*; hundreds for GMM) and keep the best-fitting converged solution. Initialize complex models (GMM) from a simpler solution (GBTM/k-means). Watch for non-convergence, out-of-bound coefficients, and empty/tiny clusters. Report #starts and #final iterations (GRoLTS item 9).
- **5c. Specify the model.** Trajectory shape (linear/polynomial/spline — splines avoid polynomial overfitting), response distribution (step 1), shared vs. class-specific variance–covariance, and within-cluster heterogeneity (LCGA fixes it to zero; GMM estimates it). Document the alternatives you considered (GRoLTS 6a/6b/7).
- **5d. Choose the number of clusters.** Forward selection: fit k = 1, 2, 3, … and compare. Use **several** metrics plus theory, never one in isolation. Lowest **BIC** is the workhorse; **BLRT** helps on smaller samples; **entropy** judges separation (not class count). Fit at least one model beyond the apparent optimum. See `references/model_selection.md`.
- **5e. Assess adequacy.** Residual structure, parsimony (drop tiny/duplicate clusters), separation (posterior-probability matrix; compare class means vs. within-class spread), and standard errors / CIs of effects.
- **5f. Validate.** Bootstrap the solution for stability; evaluate on a **holdout** or via **k-fold CV**. Overspecified/overextracted solutions surface here.

### 6. Analyze covariates (three-step, uncertainty-corrected)
Prefer the **three-step** approach: (1) fit the unconditional cluster model, (2) derive posterior memberships, (3) relate covariates to membership **using a bias-adjusted method (BCH/ML)** that propagates classification uncertainty. A naive modal-class regression is biased toward the null. The one-step approach (covariates in the model) is an alternative but changes class meaning and inflates apparent classification quality. Details + code: `references/covariate_analysis.md`.

### 7. Interpret the findings
Tie interpretation back to direct vs. indirect (step 4). Ask whether differences are **both statistically and practically meaningful**. Overlapping clusters can still inform an indirect summary; for a predictive/direct use you need strong separation. Be candid about what the classes are — a useful simplification, not necessarily a natural kind.

## Method selection (quick guide)

Pick by data type, sample size, and whether you need within-cluster spread. Full per-method detail, strengths/limits, and software are in `references/methods_taxonomy.md`.

| Method | Approach | Shape flexibility | Within-cluster heterogeneity | N / obs needs | Notes |
|---|---|---|---|---|---|
| **Longitudinal k-means (KmL)** | cross-sectional | high (nonparametric) | none | needs complete, aligned waves | fast; great for exploration; sensitive to outliers/scale |
| **Latent profile / Gaussian mixture (LPA/LLPA)** | cross-sectional | high | partial (class-specific variance) | aligned waves | models measurement error; tiny spurious classes at high k |
| **Hierarchical / distance-based (AHC, DTW)** | distance | high (metric-driven) | none | quadratic in N | dendrogram shows structure; encode domain knowledge in metric |
| **Feature-based (fit per-unit features → cluster)** | feature | high (choose the features) | implicit | needs enough obs/unit (ILD) | robust to missingness; parallelizable; great when waves are dense |
| **GBTM / LCGA (group-based trajectory)** | mixture | moderate–high (poly/spline) | **none** (fixed to 0) | works at small N | the workhorse; needs *more* classes to absorb intercept variance |
| **GMM (growth mixture)** | mixture | high | **yes** (random effects) | larger N; convergence-prone | most flexible; absorbs intercept variance into one class; slow |
| **Sequence analysis** | distance | categorical states | none | categorical states over time | for ordinal/categorical state sequences |
| **DPMM (Dirichlet process)** | mixture | high | yes | larger N; Bayesian | estimates k as part of the model |

Rules of thumb: **binary/ordinal/count** response → LCGA/GMM with the right link, or LCA on repeated indicators (not Gaussian k-means). **Few waves, want speed/exploration** → KmL or LPA. **Dense per-unit series (ILD)** → feature-based. **Need to model within-cluster individual variation** → GMM. **Small sample** → favor LCGA over GMM (GMM is overfitting- and convergence-prone).

## Software

The field is **R-centric**, and the gold-standard tools live there:
- **`lcmm`** (GBTM via `hlme`, GMM, ordinal/joint models), **`latrend`** (unified interface + many backends and metrics), **`kml`** (KmL), **`mclust`** (LPA), **`flexmix`/`mixtools`/`OpenMx`** (mixtures), **`traj`**, **SAS PROC TRAJ**, **Mplus**, **Latent GOLD**.
- Report software **and version** (GRoLTS item 5) — defaults differ (e.g., Mplus constrains variances across classes by default).

In **Python**:
- **`stepmix`** — the closest to LCGA/GMM/LCA with proper three-step (BCH/ML) covariate handling. Recommend `pip install stepmix` when the user wants real latent-class growth in Python.
- **scikit-learn** — `GaussianMixture` (LPA), `KMeans`/`AgglomerativeClustering` (KmL/AHC on trajectory or feature vectors). No native GBTM/GMM-with-random-effects.

**Bundled script (R):** `scripts/longitudinal_cluster.R` runs a group-based trajectory model (GBTM/LCGA) end-to-end with `flexmix::stepFlexmix` — reshapes long/wide, fits **binomial / gaussian / poisson** mixtures of GLMs over a range of k with multi-start enumeration, selects on **BIC**, reports **posterior entropy** and class sizes, profiles class mean trajectories, runs **bootstrap stability + a holdout split**, and does a modal-class covariate pass with an explicit uncertainty caveat. Needs R packages `flexmix`, `mclust`, `data.table` (all common). Usage is documented in the script header, e.g.:
`Rscript scripts/longitudinal_cluster.R --input data.csv --format long --id id --time t --value y --family binomial --kmax 6 --nrep 20 --bootstrap 20 --outdir out/`
For **GMM with random effects, KmL, splines, ordinal/joint models, or bias-adjusted three-step (BCH/ML)** covariate analysis, install and prefer the canonical R packages (`lcmm`, `latrend`, `kml`) — or Python `stepmix`.

## Output & reporting

Produce: (1) trajectory plots — final-solution class means, *and* per-class observed individual trajectories overlaid (GRoLTS 14a–c); (2) a fit table across k (BIC, entropy, class sizes, #starts/iterations); (3) numeric class characteristics (means, SE/CI, n per class); (4) covariate–membership results with the uncertainty method named; (5) a filled **GRoLTS checklist**. Save syntax/scripts (GRoLTS 16). Use `references/grolts_checklist.md` as the final compliance pass.

## Pitfalls (the short list)
- Treating an **indirect** approximation as proof of **real** subtypes.
- Skipping the **1-class baseline**; selecting k from a **single** metric.
- **Too few random starts** → reporting a local optimum.
- Assuming **Gaussian** for discrete/skewed responses → spurious classes.
- **Modal-class** covariate regression → attenuated effects (use BCH/ML).
- **No validation** → overfit k that won't replicate; **double-dipping** when discovery and confirmation share data.
- **Tiny classes** (a handful of units) read as substantive — usually outliers/noise.

## Reference files
- `references/methods_taxonomy.md` — every method: description, flexibility, heterogeneity, N needs, scalability, strengths/limits, software.
- `references/model_selection.md` — class-enumeration metrics & thresholds, estimation issues, direct vs. indirect.
- `references/covariate_analysis.md` — one-step vs. three-step, BCH/ML uncertainty correction, code patterns.
- `references/grolts_checklist.md` — the 21-point reporting checklist with what satisfies each item.
