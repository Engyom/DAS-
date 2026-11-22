"""T10: gait sensitivity study on SIMULATED passes (research plan 6.1).

One-at-a-time sweeps around a baseline, plus one 2-D grid (noise x lane):
  channel spacing  0.25, 0.4, 0.5, 1, 2 m      (at lanes 0.5 / 1 / 2 m)
  noise level      x0.5 ... x16 of the generator's default
  lane distance    0.5 ... 3 m
  cadence          90 ... 140 steps/min (fixed per pass)
Baseline: spacing 0.4 m, gauge length 0.8 m, lane 1 m, default noise,
cadence drawn from the generator's 95-125 steps/min.

What is measured per setting (DAS vs the TRUE simulated footsteps, so the
numbers are sensor + pipeline error only, no reference-system error):
  detection recall / precision at +/-50 ms after the median offset,
  timing SD, QC pass rate (the v2.1 QC: >= 8 validated steps, fit R^2 >= 0.95,
  inliers >= 80 %), stride-length bias / SD / RMSE, cadence and speed error SD,
  and a detection SNR [dB] computed with the pipeline's own band-pass and
  energy envelope (median envelope at detected footsteps / noise floor), so
  the same number can be measured on Phase 0 recordings and the envelope
  read off at the real noise level (task T16).

Generator limits that bound every result here (section 6.1): white noise,
ideal moving-average gauge, one homogeneous ground, constant amplitude
model. Gauge length is not swept: in this generator it does not change the
output noise. For the spacing sweep the gauge length is the smallest whole
multiple of the spacing that is >= 0.8 m (0.25 -> 1.0, 0.4 -> 0.8,
0.5 -> 1.0, 1 -> 1.0, 2 -> 2.0 m). With gauge length = n x spacing the
generator's per-channel output noise is the same at every spacing, as the
plan requires, and MATLAB and Python average the same number of channels.

    python -m gait_eval.sensitivity --out gait_eval/results_sensitivity_v2.2
"""
import argparse, json, math, os
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from das_height.features import MIN_FILL_RATIO, extract_pass_features, gait_plausible
from das_height.footsteps import bandpass, detect_footsteps, noise_floor
from gait_eval.matching import detection_metrics, estimate_offset, match_footsteps
from gait_eval.sim_truth import simulate_pass_with_truth
from tests import sim_py

LABEL = "SIMULATED"
BASE = dict(dx=0.4, gl=0.8, lane=1.0, noise_mult=1.0, cadence=None)
TOL = 0.050
# v2.2 pipeline: dominance filter in detect_footsteps (default) + plausibility QC.
# results_sensitivity/ was produced with v2.1 (commit d50bd08); results_sensitivity_v2.2/ with this.
PLAUSIBILITY_QC = True
VERSION = "v2.2"


@contextmanager
def generator(dx, noise_mult=1.0, cadence=None):
    """Temporarily set spacing, noise and (optionally) a fixed cadence in the
    generator mirror; always restored."""
    P, G = sim_py.P, sim_py.G
    old = (P["dx"], P["noise"], G["cad_range"])
    P["dx"], P["noise"] = dx, old[1] * noise_mult
    if cadence is not None:
        G["cad_range"] = (cadence, cadence)
    try:
        yield
    finally:
        P["dx"], P["noise"], G["cad_range"] = old


def gauge_for(dx, gl_min=0.8):
    """Smallest whole multiple of dx that is >= gl_min."""
    n = max(1, math.ceil(gl_min / dx - 1e-9))
    return round(n * dx, 6)


def detection_snr_db(rec, times, band=(2.0, 60.0), win_s=0.1):
    """Median energy envelope at the detected footsteps over the noise floor,
    with the pipeline's own band-pass and window [dB]."""
    if len(times) == 0:
        return np.nan
    x = bandpass(rec.data, rec.fs, *band)
    E = uniform_filter1d(x ** 2, size=max(1, int(round(win_s * rec.fs))), axis=0)
    env = E.max(axis=1)
    idx = np.clip(np.round(np.asarray(times) * rec.fs).astype(int), 0, env.size - 1)
    return float(10 * np.log10(np.median(env[idx]) / noise_floor(E)))


def run_setting(n_passes, rng, dx, gl, lane, noise_mult=1.0, cadence=None):
    """Simulate n_passes walkers at one setting; return per-pass rows."""
    rows, det_all = [], []
    with generator(dx, noise_mult, cadence):
        for i in range(n_passes):
            H = rng.uniform(1.50, 1.90)
            k = 0.415 + 0.02 * rng.standard_normal()
            rec, truth = simulate_pass_with_truth(H, k, lane, gl, rng)
            det = detect_footsteps(rec)
            f = extract_pass_features(det["times"], det["positions"])
            det_all.append((det, truth))
            qc = (f["n_steps"] >= 8) and (f["fit_r2"] >= 0.95) and (f["inlier_frac"] >= 0.8) \
                and np.isfinite(f["step_length_m"])
            if PLAUSIBILITY_QC:                      # v2.2 QC (features.gait_plausible)
                qc = qc and gait_plausible(f["cadence_spm"], f["step_length_m"]) \
                    and f["fill_ratio"] >= MIN_FILL_RATIO
            rows.append({"pass": i, "qc_pass": bool(qc), "n_det": int(det["times"].size),
                         "snr_db": detection_snr_db(rec, det["times"]),
                         "stride_err_cm": 100 * 2 * (f["step_length_m"] - truth["step_length_m"]) if qc else np.nan,
                         "cadence_err_spm": f["cadence_spm"] - truth["cadence_spm"] if qc else np.nan,
                         "speed_err_cms": 100 * (f["speed_mps"] - truth["speed_mps"]) if qc else np.nan})
    # offset from all passes of the setting (a per-lane dry-run estimate in the real study)
    offs = [estimate_offset(d["times"], t["times"])[0] for d, t in det_all]
    offs = [o for o in offs if np.isfinite(o)]
    off = float(np.median(offs)) if offs else 0.0
    n_gt = n_det = n_tp = 0
    dts, loc = [], []
    for (d, t), r in zip(det_all, rows):
        m = detection_metrics(d["times"], t["times"], off, TOL)
        n_gt += m["n_gt"]; n_det += m["n_det"]; n_tp += m["n_matched"]
        mm = match_footsteps(d["times"], t["times"], off, TOL)
        dts.extend(mm["dt"].tolist())
        if len(mm["pairs"]):
            e = d["positions"][mm["pairs"][:, 0]] - t["positions"][mm["pairs"][:, 1]]
            loc.extend((e - np.median(e)).tolist())      # SD only: absolute offset is a generator artefact
    return rows, {"offset_ms": 1000 * off, "recall": n_tp / n_gt if n_gt else np.nan,
                  "precision": n_tp / n_det if n_det else np.nan,
                  "timing_sd_ms": 1000 * float(np.std(dts, ddof=1)) if len(dts) > 1 else np.nan,
                  "loc_sd_cm": 100 * float(np.std(loc, ddof=1)) if len(loc) > 1 else np.nan}


def summarise(rows, det):
    df = pd.DataFrame(rows)
    ok = df[df.qc_pass]
    se = ok.stride_err_cm
    return {"n_passes": len(df), "qc_pass_rate": float(df.qc_pass.mean()),
            "snr_db_median": float(df.snr_db.median()), **det,
            "stride_bias_cm": float(se.mean()) if len(se) else np.nan,
            "stride_sd_cm": float(se.std(ddof=1)) if len(se) > 1 else np.nan,
            "stride_rmse_cm": float(np.sqrt((se ** 2).mean())) if len(se) else np.nan,
            "cadence_err_sd_spm": float(ok.cadence_err_spm.std(ddof=1)) if len(ok) > 1 else np.nan,
            "speed_err_sd_cms": float(ok.speed_err_cms.std(ddof=1)) if len(ok) > 1 else np.nan}


def sweeps(n_passes=12, seed=7):
    """All one-at-a-time sweeps and the noise x lane grid. Each setting uses
    its own seeded generator, so any single setting can be re-run alone."""
    plan = []
    for dx in (0.25, 0.4, 0.5, 1.0, 2.0):
        for lane in (0.5, 1.0, 2.0):
            plan.append(("spacing", dict(dx=dx, gl=gauge_for(dx), lane=lane)))
    for nm in (0.5, 1, 2, 4, 8, 16):
        plan.append(("noise", dict(BASE, noise_mult=nm)))
    for lane in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        plan.append(("lane", dict(BASE, lane=lane)))
    for cad in (90, 100, 110, 120, 130, 140):
        plan.append(("cadence", dict(BASE, cadence=float(cad))))
    for nm in (1, 2, 4, 8, 16):
        for lane in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            plan.append(("grid", dict(BASE, noise_mult=nm, lane=lane)))
    out = []
    for j, (study, s) in enumerate(plan):
        s = {**BASE, **s}
        rng = np.random.default_rng([seed, j])
        rows, det = run_setting(n_passes, rng, s["dx"], s["gl"], s["lane"], s["noise_mult"], s["cadence"])
        out.append({"study": study, "dx_m": s["dx"], "gauge_length_m": s["gl"], "lane_m": s["lane"],
                    "noise_mult": s["noise_mult"], "cadence_spm": s["cadence"], **summarise(rows, det)})
    return pd.DataFrame(out)


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    sp = df[df.study == "spacing"]
    for lane, mk in ((0.5, "o"), (1.0, "s"), (2.0, "^")):
        s = sp[sp.lane_m == lane]
        ax[0, 0].plot(s.dx_m, s.stride_sd_cm, marker=mk, label=f"lane {lane} m")
        ax[0, 1].plot(s.dx_m, 100 * s.recall, marker=mk, label=f"lane {lane} m")
    ax[0, 0].set(xscale="log", xlabel="Channel spacing [m]", ylabel="Stride error SD [cm]", title="Stride error vs channel spacing")
    ax[0, 1].set(xscale="log", xlabel="Channel spacing [m]", ylabel="Recall at +/-50 ms [%]", title="Footstep detection vs channel spacing")
    for a in ax[0, :2]:
        a.minorticks_off(); a.set_xticks([0.25, 0.4, 0.5, 1, 2]); a.set_xticklabels(["0.25", "0.4", "0.5", "1", "2"]); a.legend(fontsize=8)
    ln = df[df.study == "lane"]
    ax[0, 2].plot(ln.lane_m, ln.stride_sd_cm, "o-", label="stride error SD [cm]")
    ax[0, 2].plot(ln.lane_m, 100 * (1 - ln.recall), "s--", label="missed footsteps [%]")
    ax[0, 2].set(xlabel="Lane distance [m]", title="Lane distance (baseline noise)"); ax[0, 2].legend(fontsize=8)
    nz = df[df.study == "noise"]
    ax[1, 0].plot(nz.snr_db_median, nz.stride_sd_cm, "o-", label="stride error SD [cm]")
    ax[1, 0].plot(nz.snr_db_median, 100 * nz.qc_pass_rate, "s--", label="QC pass rate [%]")
    for _, r in nz.iterrows():
        ax[1, 0].annotate(f"x{r.noise_mult:g}", (r.snr_db_median, r.stride_sd_cm), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1, 0].set(xlabel="Detection SNR [dB] (lane 1 m)", title="Noise level (x0.5 = cleaner: filter side-lobe failure)"); ax[1, 0].legend(fontsize=8)
    cd = df[df.study == "cadence"]
    ax[1, 1].plot(cd.cadence_spm, 100 * cd.recall, "o-", label="recall [%]")
    ax[1, 1].plot(cd.cadence_spm, 100 * cd.precision, "s--", label="precision [%]")
    ax2 = ax[1, 1].twinx(); ax2.plot(cd.cadence_spm, cd.stride_sd_cm, "^:", color="C2", label="stride SD [cm]")
    ax2.set_ylabel("Stride error SD [cm]")
    ax[1, 1].set(xlabel="Cadence [steps/min]", ylabel="%", title="Cadence"); ax[1, 1].legend(fontsize=8, loc="lower left")
    g = df[df.study == "grid"].pivot(index="noise_mult", columns="lane_m", values="stride_rmse_cm")
    q = df[df.study == "grid"].pivot(index="noise_mult", columns="lane_m", values="qc_pass_rate")
    im = ax[1, 2].imshow(np.where(q.values >= 0.8, g.values, np.nan), cmap="viridis_r", aspect="auto", origin="lower", vmin=0, vmax=8)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            txt = "QC<80%" if q.values[i, j] < 0.8 else f"{g.values[i, j]:.1f}"
            ax[1, 2].text(j, i, txt, ha="center", va="center", fontsize=7, color="k" if q.values[i, j] < 0.8 else "w")
    ax[1, 2].set_xticks(range(g.shape[1])); ax[1, 2].set_xticklabels([f"{c:g}" for c in g.columns])
    ax[1, 2].set_yticks(range(g.shape[0])); ax[1, 2].set_yticklabels([f"x{r:g}" for r in g.index])
    ax[1, 2].set(xlabel="Lane distance [m]", ylabel="Noise (x default)", title="Operating envelope: stride RMSE [cm]")
    fig.colorbar(im, ax=ax[1, 2], shrink=0.8)
    fig.suptitle(f"SIMULATED - T10 gait sensitivity, pipeline {VERSION} (conditional on the generator's white-noise, ideal-gauge model; not real-world accuracy)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=125); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_sensitivity_v2.2"))
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = sweeps(a.passes, a.seed)
    df.insert(0, "label", LABEL)
    df.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
    plot(df, os.path.join(a.out, "sensitivity.png"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"label": LABEL, "passes_per_setting": a.passes, "seed": a.seed, "baseline": BASE,
                   "note": "Sensor + pipeline error vs true simulated footsteps; conditional on the generator model."},
                  fh, indent=2)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(df.drop(columns=["label"]).round(3).to_string())
