#!/usr/bin/env Rscript
# longitudinal_cluster.R — longitudinal trajectory clustering in R.
#
# Fits group-based trajectory models (GBTM / LCGA: a mixture of GLMs over time) with
# flexmix::stepFlexmix — multi-start enumeration over k, BIC model selection, posterior
# entropy, class sizes, mean-trajectory profiles, bootstrap stability, holdout split, and
# a modal-class covariate pass (with an uncertainty caveat). Supports binary (binomial),
# continuous (gaussian), and count (poisson) responses.
#
# This is the Python-free core. For GMM WITH random effects, KmL, splines, ordinal/joint
# models, or bias-adjusted three-step (BCH/ML) covariate analysis, install and prefer the
# canonical packages:  install.packages(c("lcmm","latrend","kml"))  (see the skill refs).
#
# USAGE (long format; one row per unit x time):
#   Rscript longitudinal_cluster.R --input data.csv --format long \
#     --id idnum --time wave --value dep --group parent \
#     --family binomial --degree 2 --kmax 6 --nrep 20 \
#     --covariates poverty_cat,par_edu --bootstrap 20 --holdout 0.4 \
#     --seed 7 --outdir lca_out
#
# USAGE (wide format; trajectory columns in time order):
#   Rscript longitudinal_cluster.R --input wide.csv --format wide \
#     --id idnum --waves dep_Y1,dep_Y3,dep_Y5,dep_Y9 --family binomial

suppressMessages({library(flexmix); library(data.table); library(mclust)})

## ---- minimal arg parser ----
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag); if (length(i)) args[i + 1] else default
}
opt <- list(
  input = getarg("--input"), format = getarg("--format", "long"),
  id = getarg("--id"), time = getarg("--time"), value = getarg("--value"),
  waves = getarg("--waves"), group = getarg("--group"),
  family = getarg("--family", "auto"), degree = as.integer(getarg("--degree", "2")),
  kmax = as.integer(getarg("--kmax", "6")), nrep = as.integer(getarg("--nrep", "20")),
  covariates = getarg("--covariates", ""), bootstrap = as.integer(getarg("--bootstrap", "20")),
  holdout = as.numeric(getarg("--holdout", "0")), seed = as.integer(getarg("--seed", "0")),
  outdir = getarg("--outdir", "lca_out"))
stopifnot(!is.null(opt$input), !is.null(opt$id))
dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)
set.seed(opt$seed)
R <- c("# Longitudinal cluster analysis report (R / flexmix GBTM)\n")
say <- function(...) R[[length(R) + 1]] <<- paste0(...)

## ---- load + reshape to LONG (unit, time, value) ----
dt <- fread(opt$input)
if (opt$format == "wide") {
  wv <- strsplit(opt$waves, ",")[[1]]
  long <- melt(dt, id.vars = c(opt$id, if (!is.null(opt$group)) opt$group),
               measure.vars = wv, variable.name = "time", value.name = "value")
  long[, time := as.integer(factor(time, levels = wv))]
} else {
  long <- dt[, c(opt$id, opt$group, opt$time, opt$value), with = FALSE]
  setnames(long, c(opt$id, opt$time, opt$value), c("id_", "time", "value"),
           skip_absent = TRUE)
  if (!is.null(opt$group)) setnames(long, opt$group, "group_")
}
if (opt$format == "wide") setnames(long, opt$id, "id_")
# composite unit id = id (x group)
if (!is.null(opt$group)) {
  gcol <- if (opt$format == "wide") opt$group else "group_"
  long[, unit := paste(id_, get(gcol), sep = "__")]
} else long[, unit := as.character(id_)]
long <- long[!is.na(value)]
long[, time := as.numeric(time)]

## ---- response family ----
vals <- unique(long$value)
is_bin <- all(vals %in% c(0, 1))
fam <- opt$family
if (fam == "auto") fam <- if (is_bin) "binomial" else "gaussian"
deg <- min(opt$degree, length(unique(long$time)) - 1)  # don't exceed identifiability
say("- input: `", opt$input, "` (", opt$format, "), units: ",
    uniqueN(long$unit), ", timepoints: ", length(unique(long$time)),
    ", obs: ", nrow(long))
say("- response: ", ifelse(is_bin, "BINARY", "continuous/count"),
    " -> family: **", fam, "**, time polynomial degree: ", deg, "\n")

# flexmix binomial wants cbind(successes, failures); bare 0/1 is rejected.
if (fam == "binomial") {
  long[, nfail := 1 - value]
  fmla <- as.formula(sprintf("cbind(value, nfail) ~ poly(time, %d) | unit", deg))
} else {
  fmla <- as.formula(sprintf("value ~ poly(time, %d) | unit", deg))
}
ctrl <- list(iter.max = 300, minprior = 0)

## ---- Step 5d: enumerate k = 1..kmax (forward selection, multi-start) ----
say("## Class enumeration (forward selection; multi-start; pick on BIC + theory + validation)\n")
ent_rel <- function(post) {           # relative entropy in [0,1]; ~1 = clean separation
  k <- ncol(post); if (k < 2) return(NA_real_)
  e <- -sum(post * log(pmax(post, 1e-12)), na.rm = TRUE)
  1 - e / (nrow(post) * log(k))
}
# clusters() is per-observation; labels are constant within a unit (| unit grouping) -> dedupe
unit_classes <- function(m, data) unique(data.table(unit = data$unit, class = as.integer(clusters(m))))
step <- stepFlexmix(fmla, data = long, k = 1:opt$kmax, nrep = opt$nrep,
                    model = FLXMRglm(family = fam), control = ctrl, verbose = FALSE)
enum <- data.table(k = 1:opt$kmax)
enum[, BIC := sapply(1:opt$kmax, function(kk) {
  m <- getModel(step, as.character(kk)); tryCatch(BIC(m), error = function(e) NA_real_)})]
enum[, entropy := sapply(1:opt$kmax, function(kk) {
  m <- getModel(step, as.character(kk))
  if (kk == 1) NA_real_ else ent_rel(posterior(m))})]
enum[, min_class := sapply(1:opt$kmax, function(kk) {
  min(table(unit_classes(getModel(step, as.character(kk)), long)$class))})]
say(paste(capture.output(print(enum)), collapse = "\n"))
best_k <- enum[which.min(BIC), k]
say("\n**Suggested k = ", best_k, "** by lowest BIC. CONFIRM against theory, parsimony, ",
    "class separation (entropy), and the validation below — do not accept mechanically. ",
    "Fit at least one model beyond the optimum (done: up to k=", opt$kmax, ").\n")

best <- getModel(step, as.character(best_k))
unit_lab <- unit_classes(best, long)      # one row per unit
if (best_k > 1) say("- Entropy at k=", best_k, ": ", round(ent_rel(posterior(best)), 3),
                    " (near 1 = clean classification; compare only across similar-BIC models).")

## ---- Step 5f: validation — bootstrap stability + holdout ----
say("\n## Validation")
units <- unique(long$unit)
ref <- setNames(unit_lab$class, unit_lab$unit)
aris <- c()
for (b in seq_len(opt$bootstrap)) {
  bs <- sample(units, length(units), replace = TRUE)
  bl <- long[unit %in% bs]
  m <- tryCatch(stepFlexmix(fmla, data = bl, k = best_k, nrep = max(3, opt$nrep %/% 4),
                model = FLXMRglm(family = fam), control = ctrl, verbose = FALSE),
                error = function(e) NULL)
  if (is.null(m)) next
  bu <- unit_classes(m, bl)
  aris <- c(aris, adjustedRandIndex(ref[bu$unit], bu$class))
}
say("- Bootstrap stability (mean adjusted Rand index, ", length(aris), " resamples): **",
    round(mean(aris, na.rm = TRUE), 2), "** (sd ", round(sd(aris, na.rm = TRUE), 2),
    "). >0.6 reasonably stable; <0.4 fragile.")
if (opt$holdout > 0) {
  te <- sample(units, floor(length(units) * opt$holdout))
  say("- Holdout split: train units=", length(units) - length(te), ", test units=",
      length(te), ". Fit/choose classes on train; reserve test for confirmatory covariate ",
      "tests to avoid double-dipping.")
}

## ---- profiles: class sizes + observed mean trajectory ----
say("\n## Class profiles")
say("Class sizes (units): ", paste(sprintf("%d:%d", as.integer(names(table(unit_lab$class))),
                                            table(unit_lab$class)), collapse = "  "))
long <- merge(long, unit_lab, by = "unit", all.x = TRUE)
mt <- dcast(long[, .(m = mean(value)), by = .(class, time)], class ~ time, value.var = "m")
say("Observed mean trajectory by class (rows=class, cols=time):\n",
    paste(capture.output(print(mt)), collapse = "\n"))

## ---- covariates: modal-class pass with uncertainty caveat ----
covs <- strsplit(opt$covariates, ",")[[1]]; covs <- covs[covs != ""]
if (length(covs)) {
  cv <- unique(dt[, c(opt$id, covs), with = FALSE]); setnames(cv, opt$id, "id_")
  um <- unique(long[, .(unit, id_ = as.character(get("id_")), class)])
  cv[, id_ := as.character(get("id_"))]
  mc <- merge(um, cv, by = "id_", all.x = TRUE)
  say("\nCovariate means by class (modal assignment — UNCORRECTED for classification uncertainty):\n",
      paste(capture.output(print(mc[, lapply(.SD, mean, na.rm = TRUE), by = class,
            .SDcols = covs])), collapse = "\n"))
  say("\n> NOTE: modal-class covariate analysis ignores classification uncertainty and ",
      "ATTENUATES effects. For inference use a bias-adjusted three-step (BCH/ML) via R ",
      "`lcmm`/`latrend` or `stepmix` (Python). See references/covariate_analysis.md.")
}

## ---- figures ----
png(file.path(opt$outdir, "enumeration_and_trajectories.png"), 1100, 460, res = 110)
par(mfrow = c(1, 2))
plot(enum$k, enum$BIC, type = "b", pch = 19, xlab = "k", ylab = "BIC", main = "BIC vs k")
abline(v = best_k, col = "red", lty = 2)
tt <- sort(unique(long$time)); cols <- seq_len(best_k)
matplot(tt, t(as.matrix(mt[, -1, with = FALSE])), type = "b", pch = 19, lty = 1,
        col = cols, xlab = "time", ylab = "response", main = "Class mean trajectories")
legend("topright", legend = paste0("class ", mt$class), col = cols, lty = 1, bty = "n", cex = .8)
invisible(dev.off())

## ---- write report + assignments ----
writeLines(c(unlist(R),
    paste0("\n## Figure\n- ", file.path(opt$outdir, "enumeration_and_trajectories.png")),
    "\n## Reporting\nComplete the GRoLTS checklist (references/grolts_checklist.md) before write-up."),
    file.path(opt$outdir, "report.md"))
fwrite(unit_lab, file.path(opt$outdir, "assignments.csv"))
cat(paste(unlist(R), collapse = "\n"),
    sprintf("\n\n[wrote %s/report.md, enumeration_and_trajectories.png, assignments.csv]\n", opt$outdir))
