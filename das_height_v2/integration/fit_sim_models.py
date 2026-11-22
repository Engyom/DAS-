"""Fit the height models on ALL simulated passes and save them as JSON.

T12 (research plan): a model trained once on the 180 SIMULATED passes,
saved so the team's MATLAB system can use it without retraining. This file
is new; it imports das_height (v2.2 since 22 Nov 2026; the v2.1 models are
in Git at tag v2.1-sim-baseline).

Usage (from das_height_v2/):
    python -m integration.fit_sim_models \
        --features ../sim_scratch/features_v2.2.csv \
        --metrics  ../sim_scratch/results_v2.2/metrics.json \
        --out      integration/models

Two models are saved, fitted exactly as in das_height.train_eval:
  steplength_linear : height = a + b * step_length_m      (default model)
  ridge             : standardised ridge on FEATURE_NAMES, alpha by LOSO

Each JSON stores everything needed to predict with plain arithmetic
(no scikit-learn or pickle needed to use it), plus its provenance and the
LOSO pass-level RMSE used as an approximate uncertainty. All of it is
SIMULATED: it says nothing about accuracy on real DAS recordings.
"""
import argparse, hashlib, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from das_height.features import FEATURE_NAMES, MIN_FILL_RATIO, PLAUSIBLE_CADENCE_SPM, PLAUSIBLE_STEP_M
from das_height.train_eval import ALPHAS, qc_filter, subject_weights

PIPELINE_VERSION = "v2.2-sim"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _provenance(df, features_path, feats):
    X = df[feats]
    return {
        "data_label": "SIMULATED",
        "pipeline_version": PIPELINE_VERSION,
        "trained_on": {
            "features_csv_sha256": _sha256(features_path),
            "n_passes": int(len(df)),
            "n_subjects": int(df["subject_id"].nunique()),
            "height_cm_range": [float(df.height_cm.min()), float(df.height_cm.max())],
            "feature_range": {f: [float(X[f].min()), float(X[f].max())] for f in feats},
        },
        "qc": {"min_steps": 8, "min_r2": 0.95, "min_inliers": 0.8,
               "plausible_cadence_spm": list(PLAUSIBLE_CADENCE_SPM), "plausible_step_m": list(PLAUSIBLE_STEP_M),
               "min_fill_ratio": MIN_FILL_RATIO},
        "note": ("SIMULATED training data: height enters the simulation only through the "
                 "assumed rule step length = k x height. Not validated on real DAS data."),
    }


def fit_steplength_linear(df):
    g = df["subject_id"].astype(str).to_numpy()
    m = LinearRegression().fit(df[["step_length_m"]].to_numpy(float), df.height_cm.to_numpy(float),
                               sample_weight=subject_weights(g))
    return {"features": ["step_length_m"], "mean": [0.0], "scale": [1.0],
            "coef": [float(m.coef_[0])], "intercept": float(m.intercept_), "alpha": None}, m


def fit_ridge(df, feats=FEATURE_NAMES):
    feats = list(feats)
    X, y = df[feats].to_numpy(float), df.height_cm.to_numpy(float)
    g = df["subject_id"].astype(str).to_numpy()
    m = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                     cv=LeaveOneGroupOut(), scoring="neg_mean_absolute_error")
    m.fit(X, y, groups=g, ridge__sample_weight=subject_weights(g))
    sc, rg = m.best_estimator_.named_steps["standardscaler"], m.best_estimator_.named_steps["ridge"]
    return {"features": feats, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_),
            "alpha": float(m.best_params_["ridge__alpha"])}, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--metrics", required=True, help="metrics.json of the same SIMULATED run")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models"))
    a = ap.parse_args(argv)
    raw = pd.read_csv(a.features)
    if "error" in raw.columns:
        raw = raw[raw["error"].isna()]
    df, n_rej = qc_filter(raw)
    met = json.load(open(a.metrics))
    if met.get("data_label") != "SIMULATED":
        raise ValueError("metrics.json must come from a SIMULATED run")
    os.makedirs(a.out, exist_ok=True)
    for name, fitter in (("steplength_linear", fit_steplength_linear), ("ridge", fit_ridge)):
        params, _ = fitter(df)
        doc = {"model": name, "model_version": f"{name}@{PIPELINE_VERSION}", **params,
               "uncertainty_cm": float(met["pass_level"][name]["RMSE_cm"]),
               "uncertainty_note": ("LOSO pass-level RMSE on SIMULATED data; a typical error "
                                    "size, not a per-person uncertainty"),
               **_provenance(df, a.features, params["features"])}
        path = os.path.join(a.out, f"{name}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {path}: intercept {doc['intercept']:.3f}, coef {np.round(doc['coef'], 4).tolist()}, "
              f"alpha {doc['alpha']}, uncertainty {doc['uncertainty_cm']:.2f} cm (QC rejected {n_rej})")


if __name__ == "__main__":
    main()
