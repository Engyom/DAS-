"""estimate_height: one walking pass in, gait parameters and height out.

The team's agreed interface (11 output fields):
    height_cm, uncertainty_cm, n_steps, qc_pass, qc_reason, speed_mps,
    cadence_spm, stride_m, direction, model_version, data_label

It runs the frozen v2.1 pipeline (das_height.footsteps, das_height.features)
unchanged, then applies a model saved by integration/fit_sim_models.py.

Limits (by design, see the research plan):
  - The person must walk roughly PARALLEL to the fibre with at least
    8 validated footsteps; a person crossing the fibre fails QC and gets
    no height (height_cm = NaN). Use the camera for those.
  - The saved models are trained on SIMULATED data only. On real data
    their output is a sim-to-real test, not a validated measurement.
  - uncertainty_cm is a typical simulated error size, not per person.
"""
import json, math, os
import numpy as np

from das_height.schema import DASRecord
from das_height.footsteps import detect_footsteps
from das_height.features import extract_pass_features, _robust_line

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL = "steplength_linear"
LABELS = ("SIMULATED", "EXPERIMENTAL")
UNITS = ("strain", "strain_rate")
OUTPUT_FIELDS = ("height_cm", "uncertainty_cm", "n_steps", "qc_pass", "qc_reason", "speed_mps",
                 "cadence_spm", "stride_m", "direction", "model_version", "data_label")
_CACHE = {}


def load_model(name=DEFAULT_MODEL, model_dir=MODEL_DIR):
    key = (name, model_dir)
    if key not in _CACHE:
        path = os.path.join(model_dir, f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file not found: {path} (run integration.fit_sim_models)")
        with open(path) as f:
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def predict_height(model, features):
    """Height [cm] from a saved model and a dict of feature values."""
    z = [(features[f] - m) / s for f, m, s in zip(model["features"], model["mean"], model["scale"])]
    return float(model["intercept"] + float(np.dot(model["coef"], z)))


def _num(v):
    v = float(v)
    return v if math.isfinite(v) else float("nan")


def estimate_height(data, fs, dx, gauge_length, x0=0.0, units="strain", data_label="SIMULATED",
                    model=DEFAULT_MODEL, min_steps=8, min_r2=0.95, min_inliers=0.8):
    """Estimate gait parameters and height for ONE walking pass.

    data : 2-D array (n_time, n_channel). Time must be axis 0. MATLAB is
           column-major: pass the matrix as time x channel, as it is stored
           in the simulation, and do not transpose it.
    fs, dx, gauge_length, x0 : Hz, m, m, m.
    Returns a dict with exactly OUTPUT_FIELDS. Numbers that cannot be
    computed are NaN; direction is +1 (towards increasing fibre distance),
    -1 (decreasing) or 0 (unknown).
    """
    if data_label not in LABELS:
        raise ValueError(f"data_label must be one of {LABELS}, got {data_label!r}")
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS}, got {units!r}")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError(f"data must be a 2-D (time x channel) array, got shape {x.shape}")
    fs, dx, gauge_length, x0 = float(fs), float(dx), float(gauge_length), float(x0)
    if not (fs > 0 and dx > 0 and gauge_length > 0):
        raise ValueError("fs, dx and gauge_length must be positive")
    mdl = load_model(model)

    out = {k: float("nan") for k in OUTPUT_FIELDS}
    out.update(n_steps=0, qc_pass=False, qc_reason="", direction=0,
               model_version=mdl["model_version"], data_label=data_label)

    rec = DASRecord(x, fs, dx, gauge_length, x0, units)
    det = detect_footsteps(rec)
    f = extract_pass_features(det["times"], det["positions"])
    out["n_steps"] = int(f["n_steps"])
    out["speed_mps"] = _num(f["speed_mps"])
    out["cadence_spm"] = _num(f["cadence_spm"])
    out["stride_m"] = _num(2.0 * f["step_length_m"])   # speed x stride period

    t, p = np.asarray(det["times"]), np.asarray(det["positions"])
    if t.size >= 3:
        o = np.argsort(t)
        b, _, _ = _robust_line(t[o], p[o])
        out["direction"] = int(np.sign(b[0]))

    reasons = []
    if f["n_steps"] < min_steps:
        reasons.append(f"too few validated footsteps ({f['n_steps']} < {min_steps}); "
                       "walk parallel to the fibre is required")
    if not (f["fit_r2"] >= min_r2):
        reasons.append(f"walking line fit R2 {f['fit_r2']:.3f} < {min_r2}")
    if not (f["inlier_frac"] >= min_inliers):
        reasons.append(f"inlier fraction {f['inlier_frac']:.2f} < {min_inliers}")
    missing = [n for n in mdl["features"] if not math.isfinite(float(f.get(n, float("nan"))))]
    if missing:
        reasons.append(f"features not computable: {missing}")
    if reasons:
        out["qc_reason"] = "; ".join(reasons)
        return out

    out["qc_pass"] = True
    out["height_cm"] = predict_height(mdl, f)
    out["uncertainty_cm"] = float(mdl["uncertainty_cm"])
    rng = mdl["trained_on"]["feature_range"]
    outside = [n for n in mdl["features"] if not (rng[n][0] <= f[n] <= rng[n][1])]
    out["qc_reason"] = "ok" if not outside else (
        f"ok; outside the simulated training range for {outside}: extrapolated")
    return out
