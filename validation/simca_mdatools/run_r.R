#!/usr/bin/env Rscript
# R driver for the open_nipals vs mdatools SIMCA validation.
#
# Reads the calibration/test CSVs written by run_py.py (out/*_raw_*.csv,
# exactly the same numbers open_nipals sees), fits one mdatools::simca()
# model per class per dataset (center=TRUE, scale=TRUE -- mdatools
# autoscales EACH class model by its own calibration mean/sd), and writes
# per-sample T2/Q, loadings, scores and classification counts (for three
# lim.type settings) to out/*_r.csv. Also builds a mdatools::simcam()
# multi-class model per dataset for the membership-matrix comparison.
#
# Run with:
#   R_LIBS_USER=~/R/library Rscript validation/simca_mdatools/run_r.R

suppressMessages(library(mdatools))

here <- function(...) file.path(dirname(sys.frame(1)$ofile %||% "."), ...)
`%||%` <- function(a, b) if (is.null(a)) b else a

# Resolve paths relative to this script's directory, regardless of cwd.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  normalizePath(".")
}
out_dir <- file.path(script_dir, "out")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

NCOMP <- 2
ALPHA <- 0.05
LIM_TYPES <- c("ddmoments", "jm", "chisq")

read_split <- function(name) {
  cal <- read.csv(file.path(out_dir, paste0(name, "_raw_cal.csv")))
  test <- read.csv(file.path(out_dir, paste0(name, "_raw_test.csv")))
  list(cal = cal, test = test)
}

feature_cols <- function(df) setdiff(colnames(df), c("class", "row_id"))

# ---------------------------------------------------------------------
# Fit one class model, write distances/loadings/scores, sweep lim.types
# ---------------------------------------------------------------------
process_class <- function(name, cls, split, fcols) {
  cal <- split$cal
  test <- split$test

  cls_cal <- cal[cal$class == cls, , drop = FALSE]
  Xc <- as.matrix(cls_cal[, fcols])
  Xt <- as.matrix(test[, fcols])
  c_test <- test$class

  mod <- simca(Xc, cls, ncomp = NCOMP, center = TRUE, scale = TRUE,
               alpha = ALPHA, x.test = Xt, c.test = c_test)

  # --- distances (T2, Q at ncomp=2); independent of lim.type ---
  T2_cal <- mod$calres$T2[, NCOMP]
  Q_cal <- mod$calres$Q[, NCOMP]
  T2_test <- mod$testres$T2[, NCOMP]
  Q_test <- mod$testres$Q[, NCOMP]

  dist_df <- rbind(
    data.frame(row_id = cls_cal$row_id, set = "cal", true_class = cls,
               T2_r = T2_cal, Q_r = Q_cal),
    data.frame(row_id = test$row_id, set = "test", true_class = c_test,
               T2_r = T2_test, Q_r = Q_test)
  )
  write.csv(dist_df, file.path(out_dir, sprintf("%s_%s_distances_r.csv", name, cls)),
            row.names = FALSE)

  # --- loadings ---
  loadings_df <- as.data.frame(mod$loadings[, 1:NCOMP])
  colnames(loadings_df) <- paste0("Comp", seq_len(NCOMP))
  loadings_df$feature <- fcols
  loadings_df <- loadings_df[, c("feature", paste0("Comp", seq_len(NCOMP)))]
  write.csv(loadings_df, file.path(out_dir, sprintf("%s_%s_loadings_r.csv", name, cls)),
            row.names = FALSE)

  # --- scores ---
  scores_cal <- mod$calres$scores[, 1:NCOMP, drop = FALSE]
  scores_test <- mod$testres$scores[, 1:NCOMP, drop = FALSE]
  colnames(scores_cal) <- colnames(scores_test) <- paste0("Comp", seq_len(NCOMP))
  scores_df <- rbind(
    data.frame(row_id = cls_cal$row_id, set = "cal", true_class = cls, scores_cal),
    data.frame(row_id = test$row_id, set = "test", true_class = c_test, scores_test)
  )
  write.csv(scores_df, file.path(out_dir, sprintf("%s_%s_scores_r.csv", name, cls)),
            row.names = FALSE)

  # --- classification counts, swept over lim.type ---
  counts <- list()
  for (lt in LIM_TYPES) {
    mod_lt <- setDistanceLimits(mod, lim.type = lt, alpha = ALPHA)
    calres_lt <- predict(mod_lt, Xc, c.ref = rep(cls, nrow(Xc)))
    testres_lt <- predict(mod_lt, Xt, c.ref = c_test)

    cpred_cal <- calres_lt$c.pred[, NCOMP, 1]
    cpred_test <- testres_lt$c.pred[, NCOMP, 1]

    n_accept_cal <- sum(cpred_cal == 1)
    counts[[length(counts) + 1]] <- data.frame(
      model_class = cls, lim_type = lt, eval_set = "cal", true_class = cls,
      n_total = length(cpred_cal), n_accepted = n_accept_cal
    )
    for (tc in sort(unique(c_test))) {
      idx <- which(c_test == tc)
      n_accept <- sum(cpred_test[idx] == 1)
      counts[[length(counts) + 1]] <- data.frame(
        model_class = cls, lim_type = lt, eval_set = "test", true_class = tc,
        n_total = length(idx), n_accepted = n_accept
      )
    }
  }
  list(counts = do.call(rbind, counts), model = mod)
}

run_dataset <- function(name) {
  split <- read_split(name)
  fcols <- feature_cols(split$cal)
  classes <- sort(unique(split$cal$class))

  all_counts <- list()
  models <- list()
  for (cls in classes) {
    res <- process_class(name, cls, split, fcols)
    all_counts[[length(all_counts) + 1]] <- res$counts
    models[[cls]] <- res$model
  }
  counts_df <- do.call(rbind, all_counts)
  write.csv(counts_df, file.path(out_dir, sprintf("%s_classification_counts_r.csv", name)),
            row.names = FALSE)

  # --- multi-class simcam membership matrix (default lim.type=ddmoments) ---
  mod_list <- lapply(classes, function(cls) models[[cls]])
  msm <- simcam(mod_list)

  Xc_all <- as.matrix(split$cal[, fcols])
  Xt_all <- as.matrix(split$test[, fcols])

  pred_cal <- predict(msm, Xc_all, c.ref = split$cal$class)
  pred_test <- predict(msm, Xt_all, c.ref = split$test$class)

  membership_df <- function(row_ids, true_classes, pred) {
    # pred$c.pred: [nobj, 1, nclasses] array, +1 accepted / -1 rejected
    cp <- pred$c.pred[, 1, , drop = FALSE]
    dim(cp) <- dim(pred$c.pred)[c(1, 3)]
    colnames(cp) <- classes
    df <- data.frame(row_id = row_ids, true_class = true_classes)
    for (cls in classes) {
      df[[paste0("accept_", cls)]] <- as.integer(cp[, cls] == 1)
    }
    df
  }

  mem_cal <- membership_df(split$cal$row_id, split$cal$class, pred_cal)
  mem_cal$set <- "cal"
  mem_test <- membership_df(split$test$row_id, split$test$class, pred_test)
  mem_test$set <- "test"
  mem_df <- rbind(mem_cal, mem_test)
  mem_df <- mem_df[, c("row_id", "set", "true_class",
                        paste0("accept_", classes))]
  write.csv(mem_df, file.path(out_dir, sprintf("%s_membership_r.csv", name)),
            row.names = FALSE)

  cat(sprintf("[%s] classes=%s n_cal=%d n_test=%d\n", name,
              paste(classes, collapse = ","), nrow(split$cal), nrow(split$test)))

  invisible(list(models = models, counts = counts_df))
}

# ---------------------------------------------------------------------
# Part C-ii: reproduce the documented mdatools Iris tutorial numbers
# (versicolor model, ddmoments, alpha=0.05, ncomp=2):
#   cal:  TP=24 FN=1
#   test: versicolor 25/25, setosa 0/25, virginica 4/25
# ---------------------------------------------------------------------
verify_iris_tutorial <- function(iris_counts) {
  ddm <- iris_counts[iris_counts$lim_type == "ddmoments" &
                        iris_counts$model_class == "versicolor", ]
  cat("\n=== Iris versicolor / ddmoments (tutorial check) ===\n")
  print(ddm[, c("eval_set", "true_class", "n_total", "n_accepted")],
        row.names = FALSE)

  cal_row <- ddm[ddm$eval_set == "cal", ]
  expected <- list(cal_tp = 24, test_versicolor = 25, test_setosa = 0, test_virginica = 4)
  actual <- list(
    cal_tp = cal_row$n_accepted,
    test_versicolor = ddm$n_accepted[ddm$eval_set == "test" & ddm$true_class == "versicolor"],
    test_setosa = ddm$n_accepted[ddm$eval_set == "test" & ddm$true_class == "setosa"],
    test_virginica = ddm$n_accepted[ddm$eval_set == "test" & ddm$true_class == "virginica"]
  )
  match <- identical(unlist(expected), unlist(actual))
  cat(sprintf("Matches documented tutorial numbers: %s\n", match))
  if (!match) {
    cat("expected: "); print(expected)
    cat("actual:   "); print(actual)
  }
}

iris_res <- run_dataset("iris")
verify_iris_tutorial(iris_res$counts)

wine_res <- run_dataset("wine")

cat("\nDone. Wrote R-side CSVs to", out_dir, "\n")
