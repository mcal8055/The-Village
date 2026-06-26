# Model Selection, Number of Clusters, and Estimation

## Direct vs. indirect clustering (decide this first)
- **Indirect (default, preferred):** clustering as an *approximation* to summarize a continuous spectrum of heterogeneity into a manageable number of representative groups. **No assumption that true distinct clusters exist**; there is no universally "true" k — the choice is practical, guided by objective + fit. Overlap can still be meaningful.
- **Direct (confirmatory):** testing for distinct, theory-defined clusters. Needs large separation; substantial overlap discredits it; lacks formal tests in most applications.

## Choosing the number of clusters (forward selection: fit k = 1, 2, 3, …)
Use **multiple** criteria + theory; never optimize a single index mechanically. Fit at least one model beyond the apparent optimum, and always include the **1-class** model.

**Likelihood-based:**
- **BIC** = p·log n − 2·log L̂ — the workhorse; **lower is better**; favored for class enumeration (Nylund et al. found BIC > AIC for mixtures). Sample-size-adjusted BIC (SABIC) is a common variant.
- **AIC** — lighter penalty; over-extracts more than BIC.
- **Approximate LRTs** — test whether k vs. k−1 fits better: **VLMR-LRT**, **adjusted LMR (aLMR)**, **bootstrap LRT (BLRT)**. Useful on **smaller** samples to curb over-extraction; on **large** samples they tend to flag statistically- but not practically-significant splits. **BLRT** is the most reliable but is "significant" on most empirical data — don't use any LRT as the sole decider.
- **k-fold cross-validation** — pick k by held-out likelihood.

**Separation / quality (judge *how good* a given k is, not how many):**
- **Entropy** of the posterior-probability matrix ∈ [0,1]. Near **1** = clean classification (posteriors near 0/1); near **0** = fuzzy. Validation-sample **median ≈ 0.85**. **Do not use entropy to choose k** — only to compare models of *similar* BIC. High entropy → less bias when predicting membership.
- **Posterior-probability matrix / average posterior probabilities** — well-separated classes have posteriors near 0/1.
- **Empty/over-fitted-mixture metric** (Malsiner-Walli) — fit one over-specified mixture; the number of **non-empty** classes estimates the true count.

**Distance-based / no-likelihood (KmL, AHC):**
- **Average Silhouette Width (ASW)** ∈ [−1,1]; higher = better; **> 0.5** suggests consistent structure (relax in exploration). **Calinski-Harabasz** (var ratio; outperformed BIC for LPA in one study), **Davies-Bouldin**.

**Bayesian:** WAIC, PSIS-LOO (DIC is criticized).

**Subjective:** elbow plot; manual inspection of the *variety* of patterns. Multiple metrics together are more reliable but may disagree — that disagreement is information.

**Upper bound on k** is set by prior knowledge, compute time (nonlinear scaling), convergence frequency, and sample size (each class needs enough units for stable estimates).

## Estimation issues
- **Local optima:** EM/ML mixtures are multimodal. Rerun with **many random starts** and keep the best converged fit. GRoLTS: **50–100 start sets per parameter**; GMM may need hundreds–thousands. Report #starts and #final iterations.
- **Better initialization:** seed a complex model (GMM) from a simpler solution (GBTM/k-means).
- **Convergence/invalid solutions:** non-convergence, out-of-bound coefficients, **empty/tiny clusters**. Mitigate: more iterations; constrain shared parameters (e.g., equal residual variances across classes) to reduce complexity; try different estimators.
- **Tiny classes** (a few units) are usually outliers/noise, not subtypes — penalize for parsimony.

## Validation (the step most people skip)
Parameters scale linearly with k → **overfitting on small samples**. Establish robustness by:
1. **Bootstrap stability** — re-estimate on resamples; a recurring solution (e.g., high adjusted Rand index across bootstraps) indicates stable estimation.
2. **Holdout/validation sample** — preferred; fit on train, evaluate held-out likelihood / assignment.
3. **k-fold cross-validation** — for limited samples.

This is also the antidote to **double-dipping**: if you used the data to *choose* k and the structure, don't claim confirmatory inference on the *same* data — confirm on a holdout or fresh sample.
