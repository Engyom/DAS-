"""Turn a folder of recordings into one feature table (features.csv).

Simulated data (labels stored inside each HDF5 file):
    python -m das_height.build_dataset --sim-dir sim_out --out features.csv

Real FEBUS data (labels from two CSV files):
    python -m das_height.build_dataset --febus-manifest passes.csv \
        --participants participants.csv --out features.csv

passes.csv       : file, subject_id, pass_id, lane_offset_m, speed_class,
                   footwear, x_start_m, x_stop_m   (file paths may be
                   relative to the folder of passes.csv)
participants.csv : subject_id, height_cm   (measured with a stadiometer)

For simulated data, if the files carry label_true_step_length_m, a
pipeline check (measured vs true step length) is printed and saved. That
column is for checking only; it is never a model input.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

from .footsteps import detect_footsteps
from .features import extract_pass_features


def features_for_record(rec, **detect_kw):
    fs = detect_footsteps(rec, **detect_kw)
    row = extract_pass_features(fs["times"], fs["positions"])
    row.update(rec.labels)
    row["source"] = rec.source
    row["noise_floor"] = fs["diagnostics"]["noise_floor"]
    return row


def _safe(fn, path, rows, **kw):
    try:
        row = fn()
    except Exception as e:                      # keep going; report the failure
        row = {"error": f"{type(e).__name__}: {e}"}
        print(f"  FAILED {path}: {row['error']}")
    row["file"] = os.path.basename(str(path))
    rows.append(row)


def from_sim_dir(folder, **detect_kw):
    from .io_sim import load_sim_h5
    rows = []
    paths = sorted(glob.glob(os.path.join(folder, "*.h5")))
    if not paths:
        raise FileNotFoundError(f"no .h5 files in {folder}")
    for p in paths:
        _safe(lambda: features_for_record(load_sim_h5(p), **detect_kw), p, rows)
    return pd.DataFrame(rows)


def from_febus(manifest, participants, **detect_kw):
    from .io_febus import load_febus
    m = pd.read_csv(manifest)
    base = os.path.dirname(os.path.abspath(manifest))
    p = pd.read_csv(participants)[["subject_id", "height_cm"]]
    if p.subject_id.duplicated().any():
        raise ValueError("participants.csv has duplicate subject_id values")
    rows = []
    for _, r in m.iterrows():
        path = r["file"] if os.path.isabs(r["file"]) else os.path.join(base, r["file"])
        labels = {k: r[k] for k in m.columns if k not in ("file", "x_start_m", "x_stop_m")}
        _safe(lambda: features_for_record(
            load_febus(path, x_start=r.get("x_start_m"), x_stop=r.get("x_stop_m"),
                       labels=labels), **detect_kw), path, rows)
    df = pd.DataFrame(rows).merge(p, on="subject_id", how="left")
    missing = df.height_cm.isna().sum()
    if missing:
        print(f"WARNING: {missing} passes have no height in participants.csv")
    return df


def pipeline_check(df):
    """Simulated data only: compare measured and true step length."""
    if "true_step_length_m" not in df or df.step_length_m.isna().all():
        return None
    err = 100 * (df.step_length_m - df.true_step_length_m)
    return {"passes": int(err.notna().sum()),
            "step_length_MAE_cm": float(np.nanmean(np.abs(err))),
            "step_length_bias_cm": float(np.nanmean(err)),
            "step_length_max_abs_cm": float(np.nanmax(np.abs(err)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir")
    ap.add_argument("--febus-manifest")
    ap.add_argument("--participants")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--band", nargs=2, type=float, default=(2.0, 60.0), metavar=("F1", "F2"))
    ap.add_argument("--thr-factor", type=float, default=8.0)
    a = ap.parse_args()
    kw = {"band": tuple(a.band), "thr_factor": a.thr_factor}
    if a.sim_dir:
        df = from_sim_dir(a.sim_dir, **kw)
    elif a.febus_manifest and a.participants:
        df = from_febus(a.febus_manifest, a.participants, **kw)
    else:
        ap.error("give --sim-dir, or --febus-manifest and --participants")
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} passes to {a.out}")
    chk = pipeline_check(df)
    if chk:
        print("pipeline check (SIMULATED):", json.dumps(chk))
        with open(os.path.splitext(a.out)[0] + "_pipeline_check.json", "w") as f:
            json.dump(chk, f, indent=2)


if __name__ == "__main__":
    main()
