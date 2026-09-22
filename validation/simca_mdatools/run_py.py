"""
Python driver for the open_nipals vs mdatools SIMCA validation.

Loads the two public datasets, writes the calibration/test splits to CSV
(read verbatim by run_r.R, so both languages fit on exactly the same
numbers), fits open_nipals SIMCA models in two ways:

1. "class_prescaled": a single-class SIMCA(scale=False) model fit on data
   that has already been autoscaled using that ONE class's own
   calibration mean/std (ddof=1, matching R's `sd`). This reproduces
   mdatools' per-class `simca(..., center=TRUE, scale=TRUE)` scaling
   exactly, so the resulting T2/Q values are comparable definition-for-
   definition with mdatools' $T2/$Q matrices (see REPORT.md part A).

2. "native_global": the ordinary way open_nipals SIMCA is meant to be
   used -- one global StandardScaler fit on all calibration rows
   (scale=True, the default), then per-class centering internally. This
   is what a user of the library actually gets, and is compared against
   mdatools' SIMCA-style limits and the multi-class `simcam` model.

Run with:
    PYTHONPATH=<repo>/src python3 validation/simca_mdatools/run_py.py

Does not modify anything under src/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

from open_nipals.simca.simca import SIMCA

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

N_COMPONENTS = 2
ALPHA = 0.95  # open_nipals alpha=0.95 <=> mdatools alpha=0.05 (same 95% limit)


# --------------------------------------------------------------------------
# Dataset splits
# --------------------------------------------------------------------------
def build_iris_split():
    """mdatools Iris tutorial split: odd 1-based rows (python idx 0,2,4,...)
    are calibration, even 1-based rows are test. 25 cal + 25 test per class.
    """
    data = load_iris()
    X = data.data
    y = np.array(data.target_names)[data.target]
    row_id = np.arange(len(y))

    cal_mask = (row_id % 2) == 0  # python idx 0,2,4,... == 1-based rows 1,3,5,...
    test_mask = ~cal_mask

    feature_names = [
        n.replace(" (cm)", "").replace(" ", "_") for n in data.feature_names
    ]
    return _package(X, y, row_id, cal_mask, test_mask, feature_names)


def build_wine_split():
    """Stratified split: within each class (in original row order), even
    local index -> calibration, odd local index -> test."""
    data = load_wine()
    X = data.data
    y = np.array(data.target_names)[data.target]
    row_id = np.arange(len(y))

    cal_mask = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        local = np.arange(len(idx))
        cal_mask[idx[local % 2 == 0]] = True
    test_mask = ~cal_mask

    feature_names = [n.replace("/", "_").replace(" ", "_") for n in data.feature_names]
    return _package(X, y, row_id, cal_mask, test_mask, feature_names)


def _package(X, y, row_id, cal_mask, test_mask, feature_names):
    return {
        "X": X,
        "y": y,
        "row_id": row_id,
        "cal_mask": cal_mask,
        "test_mask": test_mask,
        "feature_names": feature_names,
    }


def write_raw_csvs(name: str, split: dict):
    """Write calibration/test raw (unscaled) feature CSVs for R."""
    cols = split["feature_names"] + ["class", "row_id"]

    def _frame(mask):
        df = pd.DataFrame(split["X"][mask], columns=split["feature_names"])
        df["class"] = split["y"][mask]
        df["row_id"] = split["row_id"][mask]
        return df[cols]

    cal_df = _frame(split["cal_mask"])
    test_df = _frame(split["test_mask"])
    cal_df.to_csv(OUT / f"{name}_raw_cal.csv", index=False)
    test_df.to_csv(OUT / f"{name}_raw_test.csv", index=False)
    return cal_df, test_df


# --------------------------------------------------------------------------
# Per-class, pre-scaled single-class open_nipals models (Part A/B/C-i)
# --------------------------------------------------------------------------
def autoscale_params(X: np.ndarray):
    """Mean and std (ddof=1, matching R's `sd`) of X."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    return mean, std


def fit_class_prescaled_models(name: str, split: dict, cal_df, test_df):
    """For each class: autoscale that class's own calibration rows
    (ddof=1), apply the same transform to ALL test rows (every class),
    fit a single-class open_nipals SIMCA(scale=False) on the class's
    scaled calibration rows, and evaluate T2/Q on cal (own class only,
    matching mdatools' $calres) and on the full test set (matching
    mdatools' $res$test).
    """
    feature_names = split["feature_names"]
    classes = sorted(split["y"][split["cal_mask"]].astype(str).tolist())
    classes = sorted(set(classes))

    Xt_raw = test_df[feature_names].to_numpy(dtype=float)
    test_row_id = test_df["row_id"].to_numpy()
    test_true_class = test_df["class"].to_numpy()

    counts_rows = []

    for cls in classes:
        cls_cal_df = cal_df[cal_df["class"] == cls]
        Xc_raw = cls_cal_df[feature_names].to_numpy(dtype=float)
        cal_row_id = cls_cal_df["row_id"].to_numpy()

        mean, std = autoscale_params(Xc_raw)
        Xc_scaled = (Xc_raw - mean) / std
        Xt_scaled = (Xt_raw - mean) / std

        y_cal = np.array([cls] * Xc_scaled.shape[0], dtype=object)
        model = SIMCA(
            n_components=N_COMPONENTS,
            alpha=ALPHA,
            scale=False,
            unknown_handling="reject",
        )
        model.fit(Xc_scaled, y_cal)

        info = model.class_models_[cls]
        pca = info.pca_model

        def raw_t2_q(X_scaled):
            Xc = X_scaled - info.class_mean
            t2 = np.asarray(pca.calc_imd(input_array=Xc)).ravel()
            q = np.asarray(pca.calc_oomd(Xc, metric="QRes")).ravel()
            return t2, q

        t2_cal, q_cal = raw_t2_q(Xc_scaled)
        t2_test, q_test = raw_t2_q(Xt_scaled)

        dist_rows = []
        for rid, t2v, qv in zip(cal_row_id, t2_cal, q_cal):
            dist_rows.append(
                {"row_id": rid, "set": "cal", "true_class": cls, "T2_py": t2v, "Q_py": qv}
            )
        for rid, tc, t2v, qv in zip(test_row_id, test_true_class, t2_test, q_test):
            dist_rows.append(
                {"row_id": rid, "set": "test", "true_class": tc, "T2_py": t2v, "Q_py": qv}
            )
        pd.DataFrame(dist_rows).to_csv(
            OUT / f"{name}_{cls}_distances_py.csv", index=False
        )

        # loadings
        loadings_df = pd.DataFrame(
            pca.loadings[:, :N_COMPONENTS],
            columns=[f"Comp{i+1}" for i in range(N_COMPONENTS)],
            index=feature_names,
        )
        loadings_df.index.name = "feature"
        loadings_df.to_csv(OUT / f"{name}_{cls}_loadings_py.csv")

        # scores
        Xc_centered = Xc_scaled - info.class_mean
        Xt_centered = Xt_scaled - info.class_mean
        scores_cal = pca.transform(Xc_centered)
        scores_test = pca.transform(Xt_centered)
        score_rows = []
        for rid, sc in zip(cal_row_id, scores_cal):
            score_rows.append(
                {"row_id": rid, "set": "cal", "true_class": cls,
                 **{f"Comp{i+1}": sc[i] for i in range(N_COMPONENTS)}}
            )
        for rid, tc, sc in zip(test_row_id, test_true_class, scores_test):
            score_rows.append(
                {"row_id": rid, "set": "test", "true_class": tc,
                 **{f"Comp{i+1}": sc[i] for i in range(N_COMPONENTS)}}
            )
        pd.DataFrame(score_rows).to_csv(
            OUT / f"{name}_{cls}_scores_py.csv", index=False
        )

        # classification counts using open_nipals' own native limits
        # (SIMCA-P style: Hotelling T2 F-limit + DModX F-limit), on the
        # class-prescaled model
        membership_cal = model.get_class_membership(Xc_scaled)
        membership_test = model.get_class_membership(Xt_scaled)
        n_accept_cal = sum(1 for m in membership_cal["member_of"] if cls in m)
        counts_rows.append(
            {
                "model_class": cls,
                "eval_set": "cal",
                "true_class": cls,
                "n_total": len(y_cal),
                "n_accepted": n_accept_cal,
                "scaling": "class_prescaled",
            }
        )
        for tc in sorted(set(test_true_class.tolist())):
            idx = np.where(test_true_class == tc)[0]
            n_accept = sum(
                1 for i in idx if cls in membership_test["member_of"][i]
            )
            counts_rows.append(
                {
                    "model_class": cls,
                    "eval_set": "test",
                    "true_class": tc,
                    "n_total": len(idx),
                    "n_accepted": n_accept,
                    "scaling": "class_prescaled",
                }
            )

    return counts_rows


# --------------------------------------------------------------------------
# Native, globally-scaled multi-class open_nipals SIMCA (Part C-i / D)
# --------------------------------------------------------------------------
def fit_native_model(name: str, split: dict, cal_df, test_df):
    feature_names = split["feature_names"]
    X_cal = cal_df[feature_names].to_numpy(dtype=float)
    y_cal = cal_df["class"].to_numpy()
    X_test = test_df[feature_names].to_numpy(dtype=float)
    y_test = test_df["class"].to_numpy()
    row_id_test = test_df["row_id"].to_numpy()
    row_id_cal = cal_df["row_id"].to_numpy()

    model = SIMCA(
        n_components=N_COMPONENTS,
        alpha=ALPHA,
        scale=True,
        unknown_handling="reject",
    )
    model.fit(X_cal, y_cal)

    counts_rows = []
    classes = list(model.classes_)
    for cls in classes:
        membership_cal = model.get_class_membership(X_cal)
        membership_test = model.get_class_membership(X_test)
        idx_cal_cls = np.where(y_cal == cls)[0]
        n_accept_cal = sum(
            1 for i in idx_cal_cls if cls in membership_cal["member_of"][i]
        )
        counts_rows.append(
            {
                "model_class": cls,
                "eval_set": "cal",
                "true_class": cls,
                "n_total": len(idx_cal_cls),
                "n_accepted": n_accept_cal,
                "scaling": "native_global",
            }
        )
        for tc in sorted(set(y_test.tolist())):
            idx = np.where(y_test == tc)[0]
            n_accept = sum(
                1 for i in idx if cls in membership_test["member_of"][i]
            )
            counts_rows.append(
                {
                    "model_class": cls,
                    "eval_set": "test",
                    "true_class": tc,
                    "n_total": len(idx),
                    "n_accepted": n_accept,
                    "scaling": "native_global",
                }
            )

    # Multi-class membership matrix (Part D), test + cal
    membership_cal = model.get_class_membership(X_cal)
    membership_test = model.get_class_membership(X_test)

    def membership_df(row_ids, true_classes, membership):
        rows = []
        for rid, tc, member_of in zip(row_ids, true_classes, membership["member_of"]):
            row = {"row_id": rid, "true_class": tc}
            for cls in classes:
                row[f"accept_{cls}"] = int(cls in member_of)
            rows.append(row)
        return pd.DataFrame(rows)

    mem_cal_df = membership_df(row_id_cal, y_cal, membership_cal)
    mem_cal_df.insert(1, "set", "cal")
    mem_test_df = membership_df(row_id_test, y_test, membership_test)
    mem_test_df.insert(1, "set", "test")
    mem_df = pd.concat([mem_cal_df, mem_test_df], ignore_index=True)
    mem_df.to_csv(OUT / f"{name}_membership_py.csv", index=False)

    return counts_rows, classes


def main():
    datasets = {
        "iris": build_iris_split(),
        "wine": build_wine_split(),
    }

    all_counts = []
    for name, split in datasets.items():
        cal_df, test_df = write_raw_csvs(name, split)
        counts_prescaled = fit_class_prescaled_models(name, split, cal_df, test_df)
        counts_native, classes = fit_native_model(name, split, cal_df, test_df)

        for row in counts_prescaled + counts_native:
            row["dataset"] = name
        all_counts.extend(counts_prescaled)
        all_counts.extend(counts_native)

        print(f"{name}: classes={classes}, "
              f"n_cal={split['cal_mask'].sum()}, n_test={split['test_mask'].sum()}")

    counts_df = pd.DataFrame(all_counts)
    counts_df.to_csv(OUT / "classification_counts_py.csv", index=False)
    print("Wrote:", sorted(p.name for p in OUT.glob("*_py.csv")))


if __name__ == "__main__":
    main()
