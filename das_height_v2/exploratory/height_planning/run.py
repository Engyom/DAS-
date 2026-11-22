"""Run the Phase B height planning model (SIMULATED + ASSUMED).

    python -m exploratory.height_planning.run --out exploratory/height_planning/results
"""
import argparse, json, os
import numpy as np
import pandas as pd
from exploratory.height_planning.planning import proven_above, study

SIGMAS = (0.03, 0.05, 0.08, 0.12)
SIZES = (12, 20, 30, 40)


def main(out, reps=300, seed=2026):
    os.makedirs(out, exist_ok=True)
    rows = []
    for cv in (0.08, 0.12):                       # large-sample behaviour
        for j, su in enumerate(SIGMAS):
            r = study(2000, su, np.random.default_rng([seed, 0, int(cv * 100), j]), cadence_cv=cv)
            rows.append({"cadence_cv": cv, "sigma_u": su, **r})
    large = pd.DataFrame(rows)
    large.to_csv(os.path.join(out, "large_sample.csv"), index=False)
    prow = []
    for j, su in enumerate(SIGMAS[:3]):           # what Phase B would see
        for n in SIZES:
            rng = np.random.default_rng([seed, 1, j, n])
            res = [study(n, su, rng) for _ in range(reps)]
            d = pd.DataFrame(res)
            prow.append({"sigma_u": su, "n": n, "median_rmse_baseline": d.rmse_baseline.median(),
                         "median_rmse_A": d.rmse_A.median(), "median_rmse_B": d.rmse_B.median(),
                         "median_rmse_C": d.rmse_C.median(), "median_r_B": d.r_B.median(),
                         "p_proven_r_gt_0.45": float(np.mean([proven_above(r, n) for r in d.r_B]))})
    power = pd.DataFrame(prow)
    power.to_csv(os.path.join(out, "power.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"label": "SIMULATED + ASSUMED", "reps": reps, "seed": seed,
                   "note": "Planning model; not evidence of real-world accuracy."}, f, indent=2)
    return large, power


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    large, power = main(a.out, a.reps)
    pd.set_option("display.width", 200)
    print(large.round(3).to_string()); print(power.round(3).to_string())
