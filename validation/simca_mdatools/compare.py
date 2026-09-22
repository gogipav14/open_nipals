"""
Compare mdatools (R) vs open_nipals (Python) SIMCA outputs written by
run_r.R and run_py.py into validation/simca_mdatools/out/, and print the
comparison tables used in REPORT.md.

Run with:
    PYTHONPATH=<repo>/src python3 validation/simca_mdatools/compare.py

Requires both run_py.py and run_r.R to have already been run (in that
order, since run_r.R reads the CSVs run_py.py writes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]
WINE_CLASSES = ["class_0", "class_1", "class_2"]
DATASETS = {"iris": IRIS_CLASSES, "wine": WINE_CLASSES}


def pct(x, n):
    return f"{x}/{n} ({100 * x / n:.1f}%)" if n else "n/a"


# --------------------------------------------------------------------------
# A. Per-sample T2 / Q distance agreement
# --------------------------------------------------------------------------
def compare_distances():
    print("\n" + "=" * 78)
    print("A. Per-sample T2 (score distance) and Q (residual SS) agreement")
    print("   open_nipals: single-class SIMCA(scale=False) fit on data")
    print("   autoscaled per class by that class's own cal mean/std (ddof=1)")
    print("   mdatools:    simca(Xc, cls, ncomp=2, center=T, scale=T)")
    print("=" * 78)
    rows = []
    for dataset, classes in DATASETS.items():
        for cls in classes:
            r = pd.read_csv(OUT / f"{dataset}_{cls}_distances_r.csv")
            p = pd.read_csv(OUT / f"{dataset}_{cls}_distances_py.csv")
            m = r.merge(p, on=["row_id", "set", "true_class"], validate="one_to_one")
            assert len(m) == len(r) == len(p), (
                f"{dataset}/{cls}: row mismatch R={len(r)} Py={len(p)} merged={len(m)}"
            )

            for metric, rc, pc in [("T2", "T2_r", "T2_py"), ("Q", "Q_r", "Q_py")]:
                diff = (m[rc] - m[pc]).to_numpy()
                max_abs = np.max(np.abs(diff))
                denom = np.maximum(np.abs(m[rc].to_numpy()), 1e-12)
                max_rel = np.max(np.abs(diff) / denom)
                rows.append(
                    {
                        "dataset": dataset,
                        "class": cls,
                        "metric": metric,
                        "n": len(m),
                        "max_abs_diff": max_abs,
                        "max_rel_diff": max_rel,
                    }
                )
    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------
# B. Loadings and scores agreement (sign-aligned)
# --------------------------------------------------------------------------
def compare_loadings_scores():
    print("\n" + "=" * 78)
    print("B. Loadings and scores agreement (sign-aligned per component)")
    print("=" * 78)
    rows = []
    for dataset, classes in DATASETS.items():
        for cls in classes:
            lr = pd.read_csv(OUT / f"{dataset}_{cls}_loadings_r.csv")
            lp = pd.read_csv(OUT / f"{dataset}_{cls}_loadings_py.csv")
            lr = lr.set_index("feature")
            lp = lp.set_index("feature")
            comp_cols = [c for c in lr.columns if c.startswith("Comp")]

            signs = {}
            max_load_diff = 0.0
            for c in comp_cols:
                r_col = lr.loc[lp.index, c].to_numpy()
                p_col = lp[c].to_numpy()
                sign = 1.0 if np.dot(r_col, p_col) >= 0 else -1.0
                signs[c] = sign
                max_load_diff = max(
                    max_load_diff, np.max(np.abs(r_col - sign * p_col))
                )

            sr = pd.read_csv(OUT / f"{dataset}_{cls}_scores_r.csv")
            sp = pd.read_csv(OUT / f"{dataset}_{cls}_scores_py.csv")
            m = sr.merge(
                sp, on=["row_id", "set", "true_class"], suffixes=("_r", "_py"),
                validate="one_to_one",
            )
            max_score_diff = 0.0
            for c in comp_cols:
                diff = (m[f"{c}_r"] - signs[c] * m[f"{c}_py"]).to_numpy()
                max_score_diff = max(max_score_diff, np.max(np.abs(diff)))

            rows.append(
                {
                    "dataset": dataset,
                    "class": cls,
                    "max_abs_loading_diff": max_load_diff,
                    "max_abs_score_diff": max_score_diff,
                    "n_scores_compared": len(m),
                }
            )
    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------
# C. Classification counts by limit type
# --------------------------------------------------------------------------
def compare_classification_counts():
    print("\n" + "=" * 78)
    print("C. Classification counts by limit definition")
    print("   R lim.type in {ddmoments, jm, chisq}, alpha=0.05 (mdatools)")
    print("   open_nipals: SIMCA-P style F-based T2/DModX limits, alpha=0.95")
    print("   'class_prescaled' = per-class autoscaled (matches mdatools scaling)")
    print("   'native'          = ordinary multi-class SIMCA(scale=True)")
    print("=" * 78)
    p_all = pd.read_csv(OUT / "classification_counts_py.csv")
    for dataset in DATASETS:
        r = pd.read_csv(OUT / f"{dataset}_classification_counts_r.csv")
        p = p_all[p_all["dataset"] == dataset]

        print(f"\n--- {dataset} ---")
        pivot_rows = []
        for _, row in r.iterrows():
            pivot_rows.append(
                {
                    "model_class": row["model_class"],
                    "eval_set": row["eval_set"],
                    "true_class": row["true_class"],
                    "n_total": row["n_total"],
                    f"R:{row['lim_type']}": row["n_accepted"],
                }
            )
        rdf = pd.DataFrame(pivot_rows)
        rdf = rdf.groupby(
            ["model_class", "eval_set", "true_class", "n_total"], as_index=False
        ).first()
        # collapse duplicate lim_type columns produced per-group
        r_wide = r.pivot_table(
            index=["model_class", "eval_set", "true_class", "n_total"],
            columns="lim_type",
            values="n_accepted",
        ).reset_index()
        r_wide.columns = [
            c if c in ("model_class", "eval_set", "true_class", "n_total")
            else f"R_{c}"
            for c in r_wide.columns
        ]

        p_wide = p.pivot_table(
            index=["model_class", "eval_set", "true_class", "n_total"],
            columns="scaling",
            values="n_accepted",
        ).reset_index()
        p_wide.columns = [
            c if c in ("model_class", "eval_set", "true_class", "n_total")
            else f"open_nipals_{c}"
            for c in p_wide.columns
        ]

        merged = r_wide.merge(
            p_wide, on=["model_class", "eval_set", "true_class", "n_total"]
        )
        print(merged.to_string(index=False))


# --------------------------------------------------------------------------
# C-ii. Iris versicolor tutorial reproduction check
# --------------------------------------------------------------------------
def verify_iris_tutorial():
    print("\n" + "=" * 78)
    print("C-ii. Iris versicolor tutorial reproduction (mdatools, ddmoments)")
    print("=" * 78)
    r = pd.read_csv(OUT / "iris_classification_counts_r.csv")
    ddm = r[(r.lim_type == "ddmoments") & (r.model_class == "versicolor")]
    print(ddm[["eval_set", "true_class", "n_total", "n_accepted"]].to_string(index=False))
    expected = {
        ("cal", "versicolor"): 24,
        ("test", "versicolor"): 25,
        ("test", "setosa"): 0,
        ("test", "virginica"): 4,
    }
    for _, row in ddm.iterrows():
        key = (row["eval_set"], row["true_class"])
        if key in expected:
            match = "OK" if row["n_accepted"] == expected[key] else "DIFFERS"
            print(f"  {key}: documented={expected[key]} actual={row['n_accepted']} [{match}]")


# --------------------------------------------------------------------------
# D. Multi-class membership agreement (Wine)
# --------------------------------------------------------------------------
def compare_membership():
    print("\n" + "=" * 78)
    print("D. Multi-class membership agreement (Wine, simcam vs get_class_membership)")
    print("=" * 78)
    for dataset, classes in DATASETS.items():
        r = pd.read_csv(OUT / f"{dataset}_membership_r.csv")
        p = pd.read_csv(OUT / f"{dataset}_membership_py.csv")
        m = r.merge(p, on=["row_id", "set", "true_class"], suffixes=("_r", "_py"),
                    validate="one_to_one")
        accept_cols = [f"accept_{c}" for c in classes]

        n_total = len(m)
        n_cells = n_total * len(accept_cols)
        n_agree_cells = 0
        n_agree_rows = 0
        for _, row in m.iterrows():
            row_agree = True
            for c in accept_cols:
                if row[f"{c}_r"] == row[f"{c}_py"]:
                    n_agree_cells += 1
                else:
                    row_agree = False
            if row_agree:
                n_agree_rows += 1

        print(f"\n--- {dataset} ---")
        print(f"  samples compared: {n_total}")
        print(f"  per-class-membership cell agreement: {pct(n_agree_cells, n_cells)}")
        print(f"  full membership-vector agreement:     {pct(n_agree_rows, n_total)}")


def main():
    dist_df = compare_distances()
    ls_df = compare_loadings_scores()
    compare_classification_counts()
    verify_iris_tutorial()
    compare_membership()

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(
        "Max |T2| diff over all class/dataset combos:",
        dist_df.loc[dist_df.metric == "T2", "max_abs_diff"].max(),
    )
    print(
        "Max |Q| diff over all class/dataset combos: ",
        dist_df.loc[dist_df.metric == "Q", "max_abs_diff"].max(),
    )
    print("Max |loading| diff:", ls_df["max_abs_loading_diff"].max())
    print("Max |score| diff:  ", ls_df["max_abs_score_diff"].max())


if __name__ == "__main__":
    main()
