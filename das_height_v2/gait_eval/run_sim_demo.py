"""T9 tooling check on SIMULATED passes (research plan 5.2-5.4).

Runs the whole Phase A analysis chain on simulated data where the truth is
known, so every tool can be checked before real data exist:
  dry run -> DAS-video offset -> one-to-one matching -> detection metrics
  (tolerance sweep) -> timing and localisation -> repeated-measures
  Bland-Altman per gait quantity -> repeatability.

Design mirrors Phase A: few participants, lanes 0.5 / 1 / 2 m, 3 repetitions.
Channel spacing 0.4 m and gauge length 0.8 m (FEBUS A1-NEO as reported;
to be confirmed, task T1). Ground-truth measurement error is an ASSUMPTION
(5 ms video timing SD, 1 cm footprint SD) until the dry-run budget.

Everything here is SIMULATED. The numbers check the tools; they are not
evidence of real-world accuracy.

    python -m gait_eval.run_sim_demo --out gait_eval/results
"""
import argparse, json, os

import numpy as np
import pandas as pd

from das_height.features import extract_pass_features
from das_height.footsteps import detect_footsteps
from gait_eval.agreement import proportional_bias, repeatability, rm_bland_altman
from gait_eval.matching import TOLERANCES_S, detection_metrics, estimate_offset, match_footsteps, match_positions
from gait_eval.sim_truth import channel_spacing, reference_measurement, simulate_pass_with_truth

LABEL = "SIMULATED"
QUANTITIES = {"stride_length_m": "Stride length [m]", "step_length_m": "Step length [m]",
              "cadence_spm": "Cadence [steps/min]", "speed_mps": "Speed [m/s]"}
# Provisional thresholds, research plan 5.4 (half-width of 95 % LoA)
THRESHOLDS = {"stride_length_m": {"le1m": 0.05, "2m": 0.08}, "cadence_spm": 2.0, "speed_mps": 0.05}


def run(out, n_participants=4, lanes=(0.5, 1.0, 2.0), reps=3, dx=0.4, gl=0.8, seed=11,
        time_sd=0.005, pos_sd=0.01, tol=0.050):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    heights = np.linspace(1.55, 1.85, n_participants)
    ks = 0.415 + 0.02 * rng.standard_normal(n_participants)

    with channel_spacing(dx):
        # --- dry run: one extra person, 2 passes per lane, offset estimate only
        dry = []
        for lane in lanes:
            for _ in range(2):
                rec, truth = simulate_pass_with_truth(1.70, 0.415, lane, gl, rng)
                ref = reference_measurement(truth, rng, time_sd, pos_sd)
                det = detect_footsteps(rec)
                dry.append((det["times"], ref["times"]))
        diffs = []
        for d, g in dry:
            o, _ = estimate_offset(d, g)
            if np.isfinite(o):
                diffs.append(o)
        offset = float(np.median(diffs))

        # --- main phase
        rows, det_rows, loc = [], [], []
        for p in range(n_participants):
            pid = f"P{p + 1}"
            for lane in lanes:
                for r in range(reps):
                    rec, truth = simulate_pass_with_truth(heights[p], ks[p], lane, gl, rng)
                    ref = reference_measurement(truth, rng, time_sd, pos_sd)
                    det = detect_footsteps(rec)
                    f = extract_pass_features(det["times"], det["positions"])
                    row = {"participant": pid, "lane_m": lane, "rep": r + 1,
                           "n_det": int(det["times"].size), "n_steps_validated": f["n_steps"]}
                    for q in QUANTITIES:
                        das_q = 2 * f["step_length_m"] if q == "stride_length_m" else f[q]
                        row[f"das_{q}"] = das_q
                        row[f"ref_{q}"] = ref[q]
                        row[f"true_{q}"] = truth[q]
                    rows.append(row)
                    for t in TOLERANCES_S:
                        m = detection_metrics(det["times"], ref["times"], offset, t)
                        det_rows.append({"participant": pid, "lane_m": lane, "rep": r + 1, **m})
                    mm = match_footsteps(det["times"], ref["times"], offset, tol)
                    for e, dtv in zip(match_positions(det["positions"], ref["positions"], mm["pairs"]), mm["dt"]):
                        loc.append({"participant": pid, "lane_m": lane, "pos_err_m": e, "time_resid_s": dtv})

    passes = pd.DataFrame(rows); detections = pd.DataFrame(det_rows); loc = pd.DataFrame(loc)
    passes.to_csv(os.path.join(out, "passes.csv"), index=False)
    detections.to_csv(os.path.join(out, "detection_by_pass.csv"), index=False)

    # detection summary: per participant and overall, by tolerance
    g = detections.groupby(["tol_ms"])[["n_gt", "n_det", "n_matched"]].sum()
    det_sum = pd.DataFrame({"recall": g.n_matched / g.n_gt, "precision": g.n_matched / g.n_det}).reset_index()
    det_lane = (detections[detections.tol_ms == 1000 * tol].groupby("lane_m")[["n_gt", "n_det", "n_matched"]].sum())
    det_lane = pd.DataFrame({"recall": det_lane.n_matched / det_lane.n_gt,
                             "precision": det_lane.n_matched / det_lane.n_det}).reset_index()
    det_sum.to_csv(os.path.join(out, "detection_tolerance.csv"), index=False)
    det_lane.to_csv(os.path.join(out, "detection_by_lane.csv"), index=False)

    loc_sum = loc.groupby("lane_m").agg(pos_bias_m=("pos_err_m", "mean"), pos_sd_m=("pos_err_m", "std"),
                                        time_bias_ms=("time_resid_s", lambda s: 1000 * s.mean()),
                                        time_sd_ms=("time_resid_s", lambda s: 1000 * s.std())).reset_index()
    loc_sum.to_csv(os.path.join(out, "timing_localisation_by_lane.csv"), index=False)

    # agreement: DAS vs reference, all lanes and by lane group
    agr = []
    groups = {"all lanes": passes, "lanes <= 1 m": passes[passes.lane_m <= 1.0], "lane 2 m": passes[passes.lane_m == 2.0]}
    for q in QUANTITIES:
        for gname, df in groups.items():
            ba = rm_bland_altman(df[f"das_{q}"], df[f"ref_{q}"], df.participant)
            pb = proportional_bias(df[f"das_{q}"], df[f"ref_{q}"], df.participant)
            half = max(abs(ba["loa_low"]), abs(ba["loa_high"]))   # both limits within +/- threshold
            thr = THRESHOLDS.get(q)
            if isinstance(thr, dict):
                thr = thr["2m"] if gname == "lane 2 m" else thr["le1m"]
            agr.append({"quantity": q, "lanes": gname, "n_subjects": ba["n_subjects"], "n_passes": ba["n_pairs"],
                        "bias": ba["bias"], "bias_ci_lo": ba["bias_ci"][0], "bias_ci_hi": ba["bias_ci"][1],
                        "loa_low": ba["loa_low"], "loa_low_ci_lo": ba["loa_low_ci"][0], "loa_low_ci_hi": ba["loa_low_ci"][1],
                        "loa_high": ba["loa_high"], "loa_high_ci_lo": ba["loa_high_ci"][0], "loa_high_ci_hi": ba["loa_high_ci"][1],
                        "sd_between": ba["sd_between"], "sd_within": ba["sd_within"],
                        "prop_bias_slope": pb["slope"], "prop_bias_p": pb["p"],
                        "max_abs_loa": half, "provisional_threshold": thr,
                        "within_threshold": (half <= thr) if thr is not None else None,
                        # stricter reading: the far CI bounds of both limits inside the threshold
                        "max_abs_loa_ci": max(abs(ba["loa_low_ci"][0]), abs(ba["loa_high_ci"][1])),
                        "ci_within_threshold": (max(abs(ba["loa_low_ci"][0]), abs(ba["loa_high_ci"][1])) <= thr)
                        if thr is not None else None})
    agr = pd.DataFrame(agr); agr.to_csv(os.path.join(out, "agreement.csv"), index=False)

    rep = []
    passes["cell"] = passes.participant + "|" + passes.lane_m.astype(str)
    for q in QUANTITIES:
        for src in ("das", "ref", "true"):
            rr = repeatability(passes[f"{src}_{q}"], passes.cell)
            rep.append({"quantity": q, "source": src, **{k: (v if not isinstance(v, tuple) else None) for k, v in rr.items()},
                        "sd_within_ci_lo": rr["sd_within_ci"][0], "sd_within_ci_hi": rr["sd_within_ci"][1],
                        "rc_ci_lo": rr["rc_ci"][0], "rc_ci_hi": rr["rc_ci"][1]})
    rep = pd.DataFrame(rep).drop(columns=["sd_within_ci", "rc_ci"]); rep.to_csv(os.path.join(out, "repeatability.csv"), index=False)

    summary = {"label": LABEL, "note": "Tooling check on simulated passes; not evidence of real-world accuracy.",
               "design": {"participants": n_participants, "lanes_m": list(lanes), "reps": reps, "dx_m": dx, "gauge_length_m": gl,
                          "seed": seed, "ref_time_sd_s [Assumption]": time_sd, "ref_pos_sd_m [Assumption]": pos_sd},
               "dry_run_offset_ms": 1000 * offset,
               "detection_at_50ms": det_sum[det_sum.tol_ms == 50.0].iloc[0].drop("tol_ms").to_dict()}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    plot_ba(passes, os.path.join(out, "bland_altman_sim.png"))
    return summary, agr, det_sum, det_lane, loc_sum, rep


def plot_ba(passes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    marks = {0.5: "o", 1.0: "s", 2.0: "^"}
    for ax, (q, unit, scale) in zip(axes, [("stride_length_m", "Stride length [cm]", 100),
                                           ("cadence_spm", "Cadence [steps/min]", 1),
                                           ("speed_mps", "Speed [cm/s]", 100)]):
        das, ref = passes[f"das_{q}"] * scale, passes[f"ref_{q}"] * scale
        ba = rm_bland_altman(das, ref, passes.participant)
        for lane, mk in marks.items():
            s = passes.lane_m == lane
            ax.scatter(((das + ref) / 2)[s], (das - ref)[s], marker=mk, s=22, label=f"lane {lane} m")
        for v, ls in [(ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")]:
            ax.axhline(v, color="k", ls=ls, lw=1)
        ax.set_xlabel(f"Mean of DAS and reference, {unit}")
        ax.set_ylabel("DAS minus reference")
        u = unit.split("[")[1].rstrip("]")
        ax.set_title(f"{unit.split(' [')[0]} [{u}]: bias {ba['bias']:+.2f}, LoA [{ba['loa_low']:+.2f}, {ba['loa_high']:+.2f}]", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("SIMULATED - repeated-measures Bland-Altman (tooling check, 4 participants x 3 lanes x 3 passes)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    s, agr, ds, dl, ls, rp = run(a.out, seed=a.seed)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(json.dumps(s, indent=1, default=float)); print(ds); print(dl); print(ls)
    print(agr[["quantity", "lanes", "bias", "loa_low", "loa_high", "max_abs_loa", "max_abs_loa_ci",
               "provisional_threshold", "within_threshold", "ci_within_threshold"]])
    print(rp[["quantity", "source", "sd_within", "rc", "df"]])
