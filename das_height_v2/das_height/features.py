"""Per-pass gait features for height estimation (height only).

FEATURE_NAMES are the model inputs. QC_NAMES are quality checks used to
reject unusable passes; they are not model inputs.

v2.1 change: speed, step period, cadence, step length, step-length
variability and n_steps are all computed from the same validated footsteps
(inliers of the robust position-time fit). Before, speed used the inliers
but the step period, step-length variability and n_steps used every
detection, so a false detection between two real steps shortened the step
period and inflated step_length_cv. n_detections keeps the original count.
"""
import numpy as np

# Model inputs. speed_mps is NOT a model input: speed = step length x
# cadence / 60 exactly, so using all three makes the inputs redundant and
# the ridge coefficients unstable. It is still computed and saved.
FEATURE_NAMES = ["step_length_m", "cadence_spm", "step_length_cv"]
DESCRIPTIVE_NAMES = ["speed_mps"]
# n_detections: all footsteps detected. n_steps: those validated as inliers
# of the robust position-time fit. Every feature uses only the n_steps
# validated footsteps (v2.1).
QC_NAMES = ["n_steps", "n_detections", "fit_r2", "inlier_frac", "step_length_direct_m"]


def _robust_line(t, x):
    b = np.polyfit(t, x, 1)
    r = x - np.polyval(b, t)
    mad = np.median(np.abs(r)) + 1e-12
    ok = np.abs(r) < 3 * 1.4826 * mad + 1e-9
    if ok.sum() >= 3:
        b = np.polyfit(t[ok], x[ok], 1)
    # R^2 on inliers only: isolated false detections are reported
    # separately as the inlier fraction
    r = x[ok] - np.polyval(b, t[ok])
    ss = ((x[ok] - x[ok].mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss if ss > 0 else np.nan
    return b, r2, ok


def extract_pass_features(times, positions, min_steps=6):
    """Gait features from footstep times [s] and positions [m]."""
    out = {k: np.nan for k in FEATURE_NAMES + DESCRIPTIVE_NAMES + QC_NAMES}
    n = len(times)
    out["n_detections"] = n
    out["n_steps"] = n            # no validation possible below min_steps
    if n < min_steps:
        return out
    order = np.argsort(times)
    t, x = np.asarray(times)[order], np.asarray(positions)[order]
    b, r2, ok = _robust_line(t, x)
    # From here on every feature uses the same validated footsteps: the
    # inliers of the robust position-time fit. Rejected detections are
    # counted (n_detections - n_steps) but never used.
    t, x = t[ok], x[ok]
    out["n_steps"] = int(ok.sum())
    out["fit_r2"], out["inlier_frac"] = r2, float(ok.mean())
    if ok.sum() < min_steps:
        return out
    speed = abs(b[0])
    dt = np.diff(t)
    T0 = np.median(dt)
    T = np.median(dt[dt < 1.5 * T0])          # ignore gaps from missed steps
    consec = dt < 1.5 * T                     # consecutive footsteps only
    dl = np.abs(np.diff(x))[consec]
    out.update({
        "speed_mps": speed,
        "cadence_spm": 60.0 / T,
        "step_length_m": speed * T,           # robust: averages all steps
        "step_length_cv": float(dl.std() / dl.mean()) if dl.size > 2 else np.nan,
        "step_length_direct_m": float(np.median(dl)) if dl.size else np.nan,
    })
    return out
