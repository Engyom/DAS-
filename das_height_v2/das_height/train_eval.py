"""Train and evaluate height models with leave-one-subject-out (LOSO) CV.

Usage:
    python -m das_height.train_eval --features features.csv --out results/ --label SIMULATED

features.csv needs: subject_id, pass_id, height_cm, the FEATURE_NAMES
columns, and the QC columns (n_steps, fit_r2, inlier_frac).

Models compared (all evaluated identically):
  baseline_mean      predict the training subjects' mean height
  steplength_linear  height = a + b * step length (one feature)
  ridge              ridge on all FEATURE_NAMES; alpha by inner LOSO
  gpr                Gaussian Process Regression; gives an uncertainty

Rules that prevent leakage:
  - every outer fold holds out ALL passes of one subject;
  - scaling, alpha and GPR kernel are fitted on the training fold only;
  - training rows are weighted 1 / (passes of that subject), so a subject
    with more passes does not count more (GPR cannot take weights).
Errors are reported per pass and per subject (mean of the subject's pass
predictions), with bootstrap 95% confidence intervals over subjects.
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, MIN_FILL_RATIO, PLAUSIBLE_CADENCE_SPM, PLAUSIBLE_STEP_M

MIN_SUBJECTS = 4
ALPHAS = np.logspace(-3, 3, 13)


def model_specs(features=FEATURE_NAMES):
    """name -> (factory, feature columns, accepts sample_weight)."""
    feats = list(features)
    return {
        "baseline_mean": (lambda: DummyRegressor(strategy="mean"), feats, True),
        "steplength_linear": (lambda: LinearRegression(), ["step_length_m"], True),
        "ridge": (lambda: GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                                       {"ridge__alpha": ALPHAS}, cv=LeaveOneGroupOut(),
                                       scoring="neg_mean_absolute_error"), feats, True),
        "gpr": (lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(np.ones(len(feats)), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(1.0, noise_level_bounds=(1e-5, 1e2)),
            normalize_y=True, n_restarts_optimizer=3, random_state=0)), feats, False),
    }


def qc_filter(df, min_steps=8, min_r2=0.95, min_inliers=0.8, plausibility=True):
    """Keep usable passes. v2.2: also require walking-plausible cadence and
    step length (features.gait_plausible) and fill_ratio >= MIN_FILL_RATIO
    (no large share of missed footsteps); plausibility=False gives v2.1."""
    need = ["subject_id", "pass_id", "height_cm", "n_steps", "fit_r2", "inlier_frac"] + FEATURE_NAMES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"features table is missing columns: {missing}")
    ok = (df["n_steps"] >= min_steps) & (df["fit_r2"] >= min_r2)
    ok &= df["inlier_frac"] >= min_inliers
    ok &= df[FEATURE_NAMES + ["height_cm"]].notna().all(axis=1)
    if plausibility:
        ok &= df.cadence_spm.between(*PLAUSIBLE_CADENCE_SPM) & df.step_length_m.between(*PLAUSIBLE_STEP_M)
        if "fill_ratio" in df.columns:              # absent in v2.1 feature tables
            ok &= df.fill_ratio >= MIN_FILL_RATIO
    return df[ok].copy(), int((~ok).sum())


def subject_weights(groups):
    s = pd.Series(groups)
    return (1.0 / s.map(s.value_counts())).to_numpy()


def metrics(err):
    err = np.asarray(err, float)
    return {"MAE_cm": float(np.mean(np.abs(err))),
            "RMSE_cm": float(np.sqrt(np.mean(err ** 2))),
            "bias_cm": float(np.mean(err)),
            "max_abs_cm": float(np.max(np.abs(err)))}


def bootstrap_ci(err, n_boot=2000, seed=0):
    """95% CI of MAE and RMSE, resampling subjects with replacement."""
    err = np.asarray(err, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, err.size, size=(n_boot, err.size))
    e = err[idx]
    mae = np.mean(np.abs(e), axis=1)
    rmse = np.sqrt(np.mean(e ** 2, axis=1))
    return {"MAE_cm_95CI": [float(np.percentile(mae, 2.5)), float(np.percentile(mae, 97.5))],
            "RMSE_cm_95CI": [float(np.percentile(rmse, 2.5)), float(np.percentile(rmse, 97.5))]}


def _fit(name, model, X, y, g, w, weighted):
    if name == "ridge":
        model.fit(X, y, groups=g, ridge__sample_weight=w)
    elif weighted:
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)
    return model


def loso_evaluate(df, features=FEATURE_NAMES):
    warnings.filterwarnings("ignore", module="sklearn.gaussian_process")
    g = df["subject_id"].astype(str).to_numpy()
    if len(np.unique(g)) < MIN_SUBJECTS:
        raise ValueError(f"need at least {MIN_SUBJECTS} subjects for LOSO, got {len(np.unique(g))}")
    y = df["height_cm"].to_numpy(float)
    specs = model_specs(features)
    preds = {n: np.full(len(y), np.nan) for n in specs}
    gpr_std = np.full(len(y), np.nan)
    alphas = []
    for tr, te in LeaveOneGroupOut().split(y, y, g):
        w = subject_weights(g[tr])
        for name, (factory, cols, weighted) in specs.items():
            X = df[cols].to_numpy(float)
            m = _fit(name, factory(), X[tr], y[tr], g[tr], w, weighted)
            if name == "gpr":
                preds[name][te], gpr_std[te] = m.predict(X[te], return_std=True)
            else:
                preds[name][te] = m.predict(X[te])
            if name == "ridge":
                alphas.append(float(m.best_params_["ridge__alpha"]))

    out = df[["subject_id", "pass_id", "height_cm"]].copy()
    for name, p in preds.items():
        out[f"pred_{name}"] = p
    out["gpr_std_cm"] = gpr_std

    subj = out.groupby("subject_id").mean(numeric_only=True)
    report = {"n_passes": int(len(df)), "n_subjects": int(subj.shape[0]),
              "features": list(features), "pass_level": {}, "subject_level": {},
              "ridge_alpha_per_fold": alphas}
    base = np.abs(subj["pred_baseline_mean"] - subj.height_cm)
    for name in preds:
        report["pass_level"][name] = metrics(out[f"pred_{name}"] - out.height_cm)
        e = subj[f"pred_{name}"] - subj.height_cm
        report["subject_level"][name] = {**metrics(e), **bootstrap_ci(e)}
        if name != "baseline_mean":
            try:
                p = wilcoxon(np.abs(e), base, alternative="less").pvalue
            except ValueError:
                p = float("nan")
            report["subject_level"][name]["p_vs_baseline"] = float(p)
    z = np.abs(out.pred_gpr - out.height_cm) / out.gpr_std_cm
    report["gpr_95pct_interval_coverage"] = float(np.mean(z <= 1.96))
    report["ridge_full_fit_for_interpretation_only"] = ridge_coefficients(df, features)
    return out, report


def ridge_coefficients(df, features):
    """Ridge fitted on ALL data, coefficients in original units (cm per unit).
    For explaining the model in the report only; NOT an evaluation."""
    X = df[list(features)].to_numpy(float); y = df.height_cm.to_numpy(float)
    g = df.subject_id.astype(str).to_numpy()
    m = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                     cv=LeaveOneGroupOut(), scoring="neg_mean_absolute_error")
    m.fit(X, y, groups=g, ridge__sample_weight=subject_weights(g))
    sc, rg = m.best_estimator_.named_steps["standardscaler"], m.best_estimator_.named_steps["ridge"]
    coef = rg.coef_ / sc.scale_
    return {"alpha": float(m.best_params_["ridge__alpha"]),
            "intercept_cm": float(rg.intercept_ - np.sum(coef * sc.mean_)),
            "coef_cm_per_unit": {f: float(c) for f, c in zip(features, coef)}}


def plot(out, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subj = out.groupby("subject_id").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    lo, hi = subj.height_cm.min() - 5, subj.height_cm.max() + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.axhline(subj.pred_baseline_mean.mean(), color="0.6", lw=1, label="mean baseline")
    ax.plot(subj.height_cm, subj.pred_ridge, "o", label="ridge")
    ax.plot(subj.height_cm, subj.pred_gpr, "s", mfc="none", label="GPR")
    ax.set_xlabel("Measured height [cm]"); ax.set_ylabel("Estimated height [cm]")
    ax.set_title(f"LOSO, one point per subject ({label})"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--label", required=True, choices=["SIMULATED", "EXPERIMENTAL"],
                    help="written into every output so results are never mixed up")
    ap.add_argument("--min-steps", type=int, default=8)
    ap.add_argument("--min-r2", type=float, default=0.95)
    ap.add_argument("--min-inliers", type=float, default=0.8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = pd.read_csv(a.features)
    if "error" in raw.columns:
        raw = raw[raw["error"].isna()]
    df, n_rej = qc_filter(raw, a.min_steps, a.min_r2, a.min_inliers)
    out, rep = loso_evaluate(df)
    rep = {"data_label": a.label, "passes_rejected_by_qc": n_rej, **rep}
    out.insert(0, "data_label", a.label)
    out.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(rep, f, indent=2)
    plot(out, os.path.join(a.out, "height_scatter.png"), a.label)
    print(json.dumps({k: rep[k] for k in ("data_label", "n_subjects", "n_passes",
                                          "passes_rejected_by_qc", "subject_level")}, indent=2))


if __name__ == "__main__":
    main()
