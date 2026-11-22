"""T10 follow-up: would two candidate v2.2 changes remove the failures found?

NOT applied to the pipeline. das_height/ (v2.1) is frozen; changing it needs
approval and a new version. This script only post-processes v2.1 outputs:
  dominance filter : drop a detection whose envelope is more than 15 dB
                     below the envelope maximum within +/-0.45 s (removes
                     band-pass ringing side-lobes at high SNR)
  plausibility QC  : accept a pass only if cadence is 70-150 steps/min and
                     step length 0.3-1.0 m (rejects passes whose step period
                     was read as a multiple of the true one after missed steps)
SIMULATED; 20 passes per setting, seed 42.

    python -m gait_eval.candidate_fixes_check
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

from das_height.features import extract_pass_features
from das_height.footsteps import bandpass, detect_footsteps
from gait_eval.sensitivity import generator
from gait_eval.sim_truth import simulate_pass_with_truth

SETTINGS = [("noise x0.5, lane 1 m", 0.5, 1.0), ("noise x2, lane 3 m", 2.0, 3.0),
            ("noise x4, lane 1.5 m", 4.0, 1.5), ("baseline (x1, lane 1 m)", 1.0, 1.0)]


def dominance(rec, det, win_s=0.45, rel_db=-15.0, band=(2.0, 60.0), env_win_s=0.1):
    x = bandpass(rec.data, rec.fs, *band)
    env = uniform_filter1d(x ** 2, max(1, int(round(env_win_s * rec.fs))), axis=0).max(axis=1)
    idx = np.round(det["times"] * rec.fs).astype(int)
    w = int(win_s * rec.fs)
    keep = np.array([env[i] >= env[max(0, i - w): i + w + 1].max() * 10 ** (rel_db / 10) for i in idx], bool)
    return det["times"][keep], det["positions"][keep]


def plausible(f, cad=(70.0, 150.0), step=(0.3, 1.0)):
    return cad[0] <= f["cadence_spm"] <= cad[1] and step[0] <= f["step_length_m"] <= step[1]


def v21_qc(f):
    return f["n_steps"] >= 8 and f["fit_r2"] >= 0.95 and f["inlier_frac"] >= 0.8 and np.isfinite(f["step_length_m"])


def run(n=20, seed=42):
    out = []
    for name, nm, lane in SETTINGS:
        rng = np.random.default_rng(seed)
        res = {k: [] for k in ("v21", "v21+plaus", "dom", "dom+plaus")}
        with generator(0.4, nm):
            for _ in range(n):
                rec, tr = simulate_pass_with_truth(rng.uniform(1.5, 1.9), 0.415, lane, 0.8, rng)
                d = detect_footsteps(rec)
                for tag, (t, x) in (("v21", (d["times"], d["positions"])), ("dom", dominance(rec, d))):
                    f = extract_pass_features(t, x)
                    ok = v21_qc(f)
                    e = 200 * (f["step_length_m"] - tr["step_length_m"]) if ok else np.nan
                    res[tag].append(e)
                    res[tag + "+plaus"].append(e if ok and plausible(f) else np.nan)
        for k, v in res.items():
            v = np.array(v, float)
            out.append({"setting": name, "variant": k, "qc_pass_rate": float(np.isfinite(v).mean()),
                        "stride_rmse_cm": float(np.sqrt(np.nanmean(v ** 2))) if np.isfinite(v).any() else np.nan})
    return out


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(run())
    print("SIMULATED - candidate fixes, not applied to the pipeline")
    print(df.pivot(index="setting", columns="variant", values=["qc_pass_rate", "stride_rmse_cm"]).round(2).to_string())
