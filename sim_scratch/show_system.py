"""Show the DAS height-estimation system working on ONE simulated walking pass.

SIMULATED DATA ONLY: heights come from the assumed model step length = k x height.
This figure shows how the pipeline works; it is not a measure of real accuracy.

Panels
  1. DAS signal along the fibre over time, with the footsteps the pipeline detected
  2. Footstep position vs time: the slope is walking speed; step length = speed x step period
  3. Height estimated by ridge (LOSO) vs measured height, every subject; this one highlighted
  4. Summary for this pass

Run (from the das_height_v2 folder, with its .venv):
  .venv\\Scripts\\python.exe C:\\Users\\yomna\\Documents\\sim_scratch\\show_system.py
  .venv\\Scripts\\python.exe C:\\Users\\yomna\\Documents\\sim_scratch\\show_system.py --pass S07_d2.0_r1

Reads only; saves one PNG next to this script.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT = r"C:\Users\yomna\Documents\das_height_v2"


def load(args):
    sys.path.insert(0, args.project)
    from das_height.io_sim import load_sim_h5
    return load_sim_h5(os.path.join(args.sim_dir, args.pass_id + ".h5"))


def make_figure(rec, preds, out_png, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from das_height.footsteps import detect_footsteps, bandpass
    from das_height.features import extract_pass_features

    fs_ = detect_footsteps(rec)
    t, x = fs_["times"], fs_["positions"]
    f = extract_pass_features(t, x)
    lab = rec.labels
    sid, pid = str(lab.get("subject_id", "?")), str(lab.get("pass_id", "?"))

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"DAS height estimation, one pass: {pid}   (SIMULATED DATA, not real accuracy)",
                 fontsize=13, fontweight="bold")

    # 1. waterfall with detected footsteps
    ax1 = fig.add_subplot(2, 2, 1)
    xf = bandpass(rec.data, rec.fs, 2.0, 60.0)
    step = max(1, int(rec.fs // 100))                       # ~100 Hz for display
    img = np.abs(xf[::step])
    tt = np.arange(img.shape[0]) * step / rec.fs
    vmax = np.percentile(img, 99.5)
    ax1.imshow(img, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
               extent=[rec.positions[0], rec.positions[-1], tt[0], tt[-1]])
    ax1.plot(x, t, "o", ms=4, mfc="none", mec="cyan", label="detected footsteps (incl. false ones)")
    ax1.set_xlabel("Position along fibre [m]"); ax1.set_ylabel("Time [s]")
    ax1.set_title("1. DAS signal (|strain|) and detected footsteps")
    ax1.legend(loc="upper left", fontsize=8)

    # 2. position vs time
    ax2 = fig.add_subplot(2, 2, 2)
    # same robust fit as das_height.features: refit after rejecting outliers
    ok = np.ones(t.size, bool)
    if t.size >= 3:
        b = np.polyfit(t, x, 1)
        res = x - np.polyval(b, t)
        ok = np.abs(res) < 3 * 1.4826 * np.median(np.abs(res)) + 1e-9
        if ok.sum() >= 3:
            b = np.polyfit(t[ok], x[ok], 1)
    ax2.plot(t[ok], x[ok], "o", ms=4, label="footsteps used")
    if (~ok).any():
        ax2.plot(t[~ok], x[~ok], "x", ms=8, color="gray",
                 label=f"rejected as false detections ({(~ok).sum()})")
    if np.isfinite(f["speed_mps"]):
        ax2.plot(t, np.polyval(b, t), "r-", lw=1.5,
                 label=f"robust fit: speed {f['speed_mps']:.2f} m/s")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Position along fibre [m]")
    ax2.set_title("2. Walking speed from footstep positions")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    # 3. estimated vs measured height, all subjects
    ax3 = fig.add_subplot(2, 2, 3)
    est_txt = "no predictions file found"
    if preds is not None:
        subj = preds.groupby("subject_id").mean(numeric_only=True)
        lo, hi = subj.height_cm.min() - 5, subj.height_cm.max() + 5
        ax3.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect estimate")
        ax3.plot(subj.height_cm, subj.pred_ridge, "o", alpha=.6, label="other subjects (ridge)")
        if sid in subj.index:
            r = subj.loc[sid]
            ax3.plot(r.height_cm, r.pred_ridge, "*", ms=18, color="crimson",
                     label=f"{sid} (this walker)")
            est_txt = (f"Ridge estimate for {sid}: {r.pred_ridge:.1f} cm "
                       f"(error {r.pred_ridge - r.height_cm:+.1f} cm)")
        mae = np.mean(np.abs(subj.pred_ridge - subj.height_cm))
        ax3.set_title(f"3. Ridge, leave-one-subject-out: MAE {mae:.1f} cm (SIMULATED)")
        ax3.set_xlabel("Measured height [cm]"); ax3.set_ylabel("Estimated height [cm]")
        ax3.legend(fontsize=8); ax3.grid(alpha=.3)

    # 4. summary
    ax4 = fig.add_subplot(2, 2, 4); ax4.axis("off")
    true_L = lab.get("true_step_length_m")
    lines = [
        f"Walker: {sid}     pass: {pid}",
        f"Lane offset from fibre: {lab.get('lane_offset_m', '?')} m",
        "",
        f"Footsteps detected: {f['n_detections']}"
        f" ({f['n_detections'] - f['n_steps']} rejected as false)",
        f"Walking speed: {f['speed_mps']:.2f} m/s",
        f"Cadence: {f['cadence_spm']:.0f} steps/min",
        f"Step length measured: {100 * f['step_length_m']:.1f} cm"
        + (f"   (true {100 * true_L:.1f} cm)" if true_L else ""),
        "",
        f"Measured height: {float(lab.get('height_cm', np.nan)):.1f} cm",
        est_txt,
        "",
        "SIMULATED: height enters the data only through the",
        "assumed rule step length = k x height. Real accuracy",
        "needs FEBUS recordings of measured volunteers.",
    ]
    ax4.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10.5)
    ax4.set_title("4. Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")
    if show:
        plt.show()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_id", default="S01_d1.0_r1")
    ap.add_argument("--sim-dir", default=os.path.join(HERE, "sim_out"))
    ap.add_argument("--predictions", default=os.path.join(HERE, "results", "predictions.csv"))
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    import pandas as pd
    rec = load(args)
    preds = pd.read_csv(args.predictions) if os.path.exists(args.predictions) else None
    out = os.path.join(HERE, f"system_view_{args.pass_id}.png")
    make_figure(rec, preds, out, show=not args.no_show)


if __name__ == "__main__":
    main()
