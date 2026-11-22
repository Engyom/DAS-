"""Run the T11 identifiability study and write tables + figures.

Usage (from das_height_v2/):
    python -m exploratory.load_identifiability.run --out exploratory/load_identifiability/results

Everything is SIMULATED (generator equations) or an ASSUMPTION (swept).
No model is trained. See README.md in this folder for the interpretation.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from . import identifiability as I

LANES = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHTS = [50.0, 60.0, 70.0, 85.0, 100.0]
GAUGES = [0.8, 1.2]          # whole multiples of dx = 0.4 m
DX = 0.4
REPS = 8

# Calibration levels: spreads in natural-log amplitude units (0.10 ~ 10 %).
# coupling/speed/footwear are ASSUMPTIONS (swept, not cited); lane and
# repeatability come from the simulation and are filled in at run time.
LEVELS = {
    "L0 uncalibrated (lane unknown 0.5-2 m, free speed, own shoes)":
        dict(lane="unknown_0.5_2", coupling=0.40, speed=0.15, footwear=0.15),
    "L1 lane marked (+/-0.25 m), coupling not calibrated":
        dict(lane=0.25, coupling=0.40, speed=0.15, footwear=0.15),
    "L2 lane +/-0.1 m, coupling calibrated to 10 %, metronome speed":
        dict(lane=0.10, coupling=0.10, speed=0.05, footwear=0.10),
    "L3 same person, place, shoes, speed (backpack design)":
        dict(lane=0.05, coupling=0.0, speed=0.03, footwear=0.0),
}


def lane_curve(gl):
    rows = []
    for d in LANES:
        for rep in range(REPS):
            a, _ = I.pass_amplitude(70.0, d, gl, np.random.default_rng(1000 + rep + int(100 * d)))
            rows.append(dict(gauge_m=gl, lane_m=d, rep=rep, lnA=a))
    return pd.DataFrame(rows)


def weight_curve(gl, lane=1.0):
    rows = []
    for W in WEIGHTS:
        for rep in range(REPS):
            a, steps = I.pass_amplitude(W, lane, gl, np.random.default_rng(5000 + rep))
            rows.append(dict(gauge_m=gl, weight_kg=W, rep=rep, lnA=a,
                             step_sd=float(np.std(steps)) if np.size(steps) else np.nan))
    return pd.DataFrame(rows)


def crossing_profile(gl):
    ys = np.round(np.arange(-3.0, 3.01, 0.25), 2)
    rows = []
    for x_off in (0.0, DX / 2):                      # on a channel / midway between channels
        for y in ys:
            rows.append(dict(gauge_m=gl, along_offset_m=x_off, lateral_m=y,
                             peak=I.single_footstep_peak(20.0 + x_off, y, DX, gl)))
    df = pd.DataFrame(rows)
    df["lnA_rel"] = np.log(df.peak / df.peak.max())
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    summary = {"data_label": "SIMULATED", "dx_m": DX, "gauges_m": GAUGES,
               "note": "Amplitude proportional to weight is a generator assumption; no model trained."}
    budgets = []
    for gl in GAUGES:
        I.configure(DX, gl)
        lc, wc, cp = lane_curve(gl), weight_curve(gl), crossing_profile(gl)
        for name, df in (("lane_curve", lc), ("weight_curve", wc), ("crossing_profile", cp)):
            df.to_csv(os.path.join(a.out, f"{name}_gl{gl:g}.csv"), index=False)

        # [Sim] check: slope of lnA on lnW should be 1 (proportionality built into the generator)
        ww = wc.dropna(subset=["lnA"])
        slope = float(np.polyfit(np.log(ww.weight_kg), ww.lnA, 1)[0])
        # [Sim] repeatability: spread of per-pass lnA, same weight/lane/gauge
        sigma_rep = float(ww.groupby("weight_kg").lnA.std(ddof=1).mean())
        step_sd = float(ww.step_sd.mean())
        # [Sim] lane sensitivity: d lnA / d lane, and spread if lane unknown in 0.5-2 m
        m = lc.groupby("lane_m").lnA.mean()
        dlnA = np.gradient(m.values, m.index.values)
        slope_at_1m = float(np.interp(1.0, m.index.values, dlnA))
        grid = np.linspace(0.5, 2.0, 301)
        sigma_lane_unknown = float(np.std(np.interp(grid, m.index.values, m.values)))
        # |y| < 0.25 m excluded: a footstep exactly on the fibre makes the point-source
        # geometry singular (horizontal distance -> 0), a model artefact, not physics.
        cq = cp[cp.lateral_m.abs() >= 0.25]
        cross_range = float(cq.groupby("along_offset_m").lnA_rel.apply(lambda s: s.max() - s.min()).max())
        along_effect = float((cq[cq.along_offset_m == 0].set_index("lateral_m").lnA_rel
                              - cq[cq.along_offset_m > 0].set_index("lateral_m").lnA_rel).abs().max())
        summary[f"gl{gl:g}"] = dict(
            lnA_vs_lnW_slope=slope, sigma_repeat_ln=sigma_rep, per_footstep_sd_ln=step_sd,
            dlnA_dlane_at_1m_per_m=slope_at_1m, sigma_lane_unknown_0p5_2_ln=sigma_lane_unknown,
            lane_0p5_to_2_ln_drop=float(m.loc[0.5] - m.loc[2.0]),
            crossing_dynamic_range_ln_abs_y_ge_0p25=cross_range,
            crossing_channel_grid_effect_ln_abs_y_ge_0p25=along_effect)

        for lvl, spec in LEVELS.items():
            lane = sigma_lane_unknown if spec["lane"] == "unknown_0.5_2" else abs(slope_at_1m) * spec["lane"]
            parts = {"repeatability [Sim]": sigma_rep, "lane [Sim x protocol]": lane,
                     "coupling [Assumption]": spec["coupling"], "speed [Assumption]": spec["speed"],
                     "footwear [Assumption]": spec["footwear"]}
            s = I.total_sigma(parts)
            within = lvl.startswith("L3")      # same person: no between-person weight classes
            budgets.append(dict(gauge_m=gl, level=lvl, **{k: round(v, 3) for k, v in parts.items()},
                                sigma_total=round(s, 3),
                                min_ratio_dprime2=round(I.min_weight_ratio(s), 2),
                                classes_45_110kg=(np.nan if within else I.n_classes(s)),
                                err_60_vs_85kg=(np.nan if within else
                                                round(I.adjacent_error(I.dprime(60, 85, s)), 3))))
    bud = pd.DataFrame(budgets)
    bud.to_csv(os.path.join(a.out, "identifiability_budget.csv"), index=False)

    # Backpack planning (within subject). sigma per pass: simulated (optimistic) and swept
    bp = []
    for W in (60.0, 70.0, 85.0):
        for load in (10.0, 20.0):
            d = float(np.log((W + load) / W))
            for sig_name, sig in (("sim repeatability", summary["gl0.8"]["sigma_repeat_ln"]),
                                  ("assumed 0.05", 0.05), ("assumed 0.10", 0.10), ("assumed 0.15", 0.15)):
                bp.append(dict(body_kg=W, load_kg=load, expected_lnA_change=round(d, 3),
                               expected_pct=round(100 * (np.exp(d) - 1), 1), sigma_pass=sig_name,
                               sigma_value=round(sig, 3), passes_per_level=I.passes_needed(d, sig)))
    pd.DataFrame(bp).to_csv(os.path.join(a.out, "backpack_planning.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    make_figures(a.out)
    print(json.dumps(summary, indent=1))
    print(bud.to_string(index=False))


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for gl, mk in ((0.8, "o-"), (1.2, "s--")):
        lc = pd.read_csv(os.path.join(out, f"lane_curve_gl{gl:g}.csv")).groupby("lane_m").lnA.agg(["mean", "std"])
        ref = lc["mean"].loc[1.0]
        ax[0].errorbar(lc.index, lc["mean"] - ref, yerr=lc["std"], fmt=mk, capsize=3, label=f"GL {gl} m")
        wc = pd.read_csv(os.path.join(out, f"weight_curve_gl{gl:g}.csv")).groupby("weight_kg").lnA.agg(["mean", "std"])
        ax[1].errorbar(wc.index, wc["mean"] - wc["mean"].loc[70.0], yerr=wc["std"], fmt=mk, capsize=3, label=f"GL {gl} m")
        cp = pd.read_csv(os.path.join(out, f"crossing_profile_gl{gl:g}.csv"))
        c0 = cp[(cp.along_offset_m == 0) & (cp.lateral_m.abs() >= 0.25)]
        ax[2].plot(c0.lateral_m, c0.lnA_rel, mk, label=f"GL {gl} m")
    ax[0].set(xlabel="Distance from fibre [m]", ylabel="ln amplitude (rel. 1 m)",
              title="Lane distance effect [Sim]")
    w = np.array([50, 100]); ax[1].plot(w, np.log(w / 70), "k:", lw=1, label="ln(W/70)")
    ax[1].set(xlabel="Body weight [kg] (lane 1 m)", ylabel="ln amplitude (rel. 70 kg)",
              title="Weight effect: built into generator [Sim]")
    ax[2].set(xlabel="Lateral position while crossing [m]", ylabel="ln peak amplitude (rel. max)",
              title="One footstep along a crossing path [Sim]")
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend()
    fig.suptitle("Load identifiability, channel spacing 0.4 m - SIMULATED (amplitude proportional to weight is an assumption)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "amplitude_confounds.png"), dpi=140); plt.close(fig)

    bud = pd.read_csv(os.path.join(out, "identifiability_budget.csv"))
    b = bud[bud.gauge_m == 0.8]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    cols = [c for c in b.columns if "[" in c]
    left = np.zeros(len(b))
    for c in cols:                                     # variance contributions
        v = b[c].to_numpy() ** 2
        ax.barh(range(len(b)), v, left=left, label=c); left += v
    for i, (s, n, r) in enumerate(zip(b.sigma_total, b.classes_45_110kg, b.min_ratio_dprime2)):
        txt = (f"  sigma={s:.2f}; within one person: detectable change x{r:.2f}" if np.isnan(n)
               else f"  sigma={s:.2f}; weight classes 45-110 kg: {int(n)}")
        ax.text(left[i] * 1.01, i, txt, va="center")
    ax.set_yticks(range(len(b))); ax.set_yticklabels([l.split(" (")[0].split(" lane")[0] for l in b.level])
    ax.invert_yaxis(); ax.set_xlabel("Variance of ln amplitude (stacked contributions)")
    ax.set_title("Confound budget per calibration level, GL 0.8 m\n(SIMULATED lane and repeatability; ASSUMED coupling, speed, footwear)")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, left.max() * 1.9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "confound_budget.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
