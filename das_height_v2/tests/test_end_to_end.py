import json, subprocess, sys, os
import numpy as np
import pandas as pd
from tests.sim_py import simulate_pass
from das_height.build_dataset import features_for_record, pipeline_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_simulated_passes_to_cli(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for s in range(6):
        H = rng.uniform(1.50, 1.85); k = 0.415 + 0.01 * rng.standard_normal()
        for d in (0.5, 2.0):
            rec, L, _ = simulate_pass(H, k, d, 1.0, rng)
            rec.labels = {"subject_id": f"S{s}", "pass_id": f"S{s}_{d}",
                          "height_cm": 100 * H, "true_step_length_m": L}
            rows.append(features_for_record(rec))
    df = pd.DataFrame(rows)
    chk = pipeline_check(df)
    assert chk["step_length_MAE_cm"] < 2.0
    csv = tmp_path / "features.csv"; df.to_csv(csv, index=False)
    r = subprocess.run([sys.executable, "-m", "das_height.train_eval", "--features", str(csv),
                        "--out", str(tmp_path / "res"), "--label", "SIMULATED"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(tmp_path / "res" / "metrics.json"))
    assert rep["data_label"] == "SIMULATED" and rep["n_subjects"] == 6
    assert "true_step_length_m" not in rep["features"]
    assert (tmp_path / "res" / "height_scatter.png").exists()
    pred = pd.read_csv(tmp_path / "res" / "predictions.csv")
    assert (pred.data_label == "SIMULATED").all()
