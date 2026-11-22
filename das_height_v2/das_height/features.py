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
QC_NAMES = ["n_steps", "n_detections", "fit_r2", "inlier_frac", "step_length_direct_m", "fill_ratio"]

# v2.2 plausibility QC (T10 sensitivity study). When footsteps are missed at
# regular intervals (low SNR) the step period is read as a multiple of the
# true one, and the remaining footsteps still lie on a clean line, so the
# fit-based QC passes them (stride errors of 20-400 cm in simulation). A pass
# is usable for height only if its gait is plausible for WALKING. These are
# judgment values; running exceeds 150 steps/min, so callers should flag such
# passes, not treat them as errors.
PLAUSIBLE_CADENCE_SPM = (70.0, 150.0)
PLAUSIBLE_STEP_M = (0.3, 1.0)
# v2.2 completeness QC: fill_ratio = validated footsteps / footsteps expected
# over the covered distance (1 + span / step length). Missed footsteps that
# are NOT regular (so cadence stays plausible) still bias the step period;
# in the T10 simulation clean passes had fill_ratio 0.86-1.02 (the lowest
# at lane 0.5 m, where the robust fit rejects some real footsteps), while
# passes with large stride errors were mostly below 0.85. The two ranges
# overlap, so this gate lowers but does not remove the low-SNR risk; it is a
# judgment value to re-set on Phase A data.
MIN_FILL_RATIO = 0.85


def gait_plausible(cadence_spm, step_length_m, cadence_range=PLAUSIBLE_CADENCE_SPM,
                   step_range=PLAUSIBLE_STEP_M):
    """True when cadence and step length are inside the walking ranges."""
    try:
        c, l = float(cadence_spm), float(step_length_m)
    except (TypeError, ValueError):
        return False
    return bool(cadence_range[0] <= c <= cadence_range[1] and step_range[0] <= l <= step_range[1])


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
        "fill_ratio": float(ok.sum() / (1.0 + np.ptp(x) / (speed * T))) if speed * T > 0 else np.nan,
    })
    return out
