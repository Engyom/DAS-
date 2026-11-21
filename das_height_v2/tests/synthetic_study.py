"""Planning study on SIMULATED data (not a pytest file; not a performance claim).

Usage: python tests/synthetic_study.py [N_SUBJ] [GL] [H_LO] [H_HI] [K_SD]
  N_SUBJ subjects, gauge length GL [m], heights uniform in [H_LO, H_HI] m,
  between-person spread K_SD of the assumed step-length/height ratio.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.sim_py import simulate_pass
from das_height.build_dataset import features_for_record
from das_height.train_eval import qc_filter, loso_evaluate


def main():
    a = sys.argv[1:]
    n_subj = int(a[0]) if len(a) > 0 else 20
    GL = float(a[1]) if len(a) > 1 else 1.0
    h_lo = float(a[2]) if len(a) > 2 else 1.50
    h_hi = float(a[3]) if len(a) > 3 else 1.80
    k_sd = float(a[4]) if len(a) > 4 else 0.02
    rng = np.random.default_rng(1)
    rows, l_err, t0 = [], [], time.time()
    for s in range(n_subj):
        H = rng.uniform(h_lo, h_hi)
        k = 0.415 + k_sd * rng.standard_normal()
        for d in (0.5, 1.0, 2.0):
            for r in range(2):
                rec, L, _ = simulate_pass(H, k, d, GL, rng)
                rec.labels = {"subject_id": f"S{s:02d}", "pass_id": f"S{s:02d}_d{d}_{r}",
                              "height_cm": 100 * H, "lane_offset_m": d}
                row = features_for_record(rec)
                l_err.append(row["step_length_m"] - L)
                rows.append(row)
    df, nrej = qc_filter(pd.DataFrame(rows))
    out, rep = loso_evaluate(df)
    rep = {"data_label": "SIMULATED", "passes_rejected_by_qc": nrej,
           "step_length_MAE_cm": float(100 * np.nanmean(np.abs(l_err))),
           "runtime_s": round(time.time() - t0, 1), **rep}
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
