# Methods Taxonomy (after Den Teuling et al., 2026)

Methods are organized into four approaches of increasing model complexity. Attributes (sample-size needs, scalability) are often stated at the *approach* level.

## Approach 1 — Cross-sectional clustering
Apply a cross-sectional cluster/mixture algorithm directly to the repeated observations, treating each time point as a variable (assumes local independence → ignores temporal dependence; nonparametric trajectory; needs complete, (near-)aligned waves). Fast; good for exploration.

- **Longitudinal k-means (KmL / LKMA).** Centroids minimize within-cluster SSE. Shape: high/nonparametric. Heterogeneity: none (assumes equal within-cluster variance). Scales excellently. Sensitive to outliers, boundary cases, scaling. Variants: fuzzy c-means, trimmed k-means, k-means++. Software: R `kml`; also SPSS/SAS/Stata.
- **Latent profile analysis / Gaussian mixture (LPA / LLPA).** Mixture of multivariate normals with posterior membership; models class-specific & time-varying variance. Shape: high. Heterogeneity: partial (variances, no random effects). Spurious tiny/empty classes at high k. Software: Mplus, Latent GOLD, R `mclust` (EM).

## Approach 2 — Distance-based clustering
Cluster by pairwise dissimilarity (Euclidean, dynamic time warping for temporal offsets, piecewise-constant approx). Encode domain knowledge in the metric. Pairwise distances scale **quadratically** in N; centroid may be uninformative; some metrics need aligned data.

- **Agglomerative hierarchical (AHC).** Bottom-up merging → dendrogram showing a spectrum of shapes; choose k from the hierarchy. Linkages: average (common default), single, centroid, Ward. Software: R `stats::hclust`.

## Approach 3 — Feature-based clustering
Reduce each trajectory independently to p features/parameters, then cluster the feature vectors cross-sectionally. Robust to missingness; parallelizable; compact fixed-size representation; **generally needs ILD** (enough observations per unit) for reliable features; features unreliable for poorly-fitting trajectories.

- **Individual time series (ITS) representations.** Per-unit fit (linear intercept+slope, polynomials, splines, change-points) → cluster coefficients. E.g., anchored k-medoids (R `akmedoids`).
- **Statistical-feature clustering.** Summaries: mean/SD/skew/range, trend/cyclic/seasonal/irregular decomposition, autocorrelation, entropy, self-similarity. Markov transition probabilities + AHC/Ward. R `cluster` (k-medoids on scaled features).
- **ARMA/ARIMA / stochastic-process features.** Model the irregular component (AR/MA/ARMA; ARIMA via differencing); Kalpakis distance + k-medoids. ILD only.

## Approach 4 — Mixture modeling
Parametric group models fit a mixture with probabilistic (posterior) membership. Handles **arbitrary numbers of measurements at arbitrary times**; compact parametric cluster trajectories; relatively low sample-size needs; can relate to external variables (multinomial logistic on membership). Cost: computationally intensive; parameters scale **linearly** with k; convergence-prone (many random starts).

- **GBTM / LCGA / GBTM (group-based trajectory).** Finite homogeneous classes; units represented only by their class trajectory; **no within-class variability** (defining feature → needs *more* classes to absorb intercept spread). Polynomials or (better) cubic B-splines. Suitable at **small N**; fast; handles MAR & unequal times. Responses: normal, censored-normal, zero-inflated Poisson, logistic, beta; multivariate/joint. Software: SAS PROC TRAJ, Stata, Mplus, OpenMx, R `lcmm`/`crimCV`/`flexmix`/`mixtools`.
- **GMM (growth mixture).** GBTM + parametric (normal) random effects → models **within-class variability** (its key advantage: absorbs intercept variance into one class, freeing classes for slope etc.). Shapes: poly/fractional-poly/spline; piecewise (PGMM) with change points; multiphase via latent transitions. Non-normal random effects: skew-normal/skew-t for robustness. Slower than GBTM; many parameters; convergence problems; evidence on GMM-vs-KmL at small N is mixed. Responses: binary/categorical/ordinal/count/zero-inflated; joint survival. Software: Mplus, Latent GOLD, R `OpenMx`/`lcmm`/`mixAK`/`flexmix`/`mixtools`; Bayesian via JAGS/Stan/`brms`.
- **Dirichlet-process mixtures (DPMM).** Number of clusters grows as needed and is estimated (no post-hoc k selection). Bayesian (MCMC/EM). R `DPpackage`, `BClustLonG`.
- **Latent-class growth trees (LCGT).** Top-down recursive 2-class GBTM splits until no significant improvement; accounts for classification error; supports covariates via three-step.

## Case-study takeaways
- Methods agree at **≤4 clusters**; diverge mainly on the "best" k due to differing shape representations and within/between assumptions.
- KmL / LLPA: fast, few decisions, good for quick exploration.
- GBTM needs more classes than GMM to represent intercept variability; GMM recovers generating groups well but is costly and convergence-prone.

Reference R code: https://github.com/niekdt/demo-clustering-longitudinal-data
