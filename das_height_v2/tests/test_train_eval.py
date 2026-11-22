import numpy as np
import pandas as pd
import pytest
from das_height.train_eval import loso_evaluate, qc_filter, subject_weights


def table(n_subj=12, passes=3, noise_k=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        H = rng.uniform(150, 190)
        k = 0.415 + noise_k * rng.standard_normal()
        for p in range(passes):
            L = k * H / 100 * (1 + 0.005 * rng.standard_normal())
            cad = rng.uniform(95, 125)
            rows.append({"subject_id": f"S{s:02d}", "pass_id": f"S{s:02d}_{p}", "height_cm": H,
                         "step_length_m": L, "cadence_spm": cad, "speed_mps": L * cad / 60,
                         "step_length_cv": abs(0.03 + 0.01 * rng.standard_normal()),
                         "n_steps": 40, "fit_r2": 0.999, "inlier_frac": 1.0})
    return pd.DataFrame(rows)


def test_models_beat_baseline_when_relation_is_strong():
    out, rep = loso_evaluate(table())
    s = rep["subject_level"]
    assert s["ridge"]["MAE_cm"] < s["baseline_mean"]["MAE_cm"]
    assert s["steplength_linear"]["MAE_cm"] < s["baseline_mean"]["MAE_cm"]
    lo, hi = s["ridge"]["MAE_cm_95CI"]
    assert lo <= s["ridge"]["MAE_cm"] <= hi


def test_no_label_leakage():
    """Changing one subject's height label must not change that subject's
    own predictions: it is never in its own training fold."""
    df = table()
    out1, _ = loso_evaluate(df)
    df2 = df.copy(); df2.loc[df2.subject_id == "S03", "height_cm"] += 50
    out2, _ = loso_evaluate(df2)
    m = df.subject_id == "S03"
    for c in ["pred_baseline_mean", "pred_steplength_linear", "pred_ridge", "pred_gpr"]:
        assert np.allclose(out1.loc[m, c], out2.loc[m, c]), c


def test_subject_weights_equalise_subjects():
    w = subject_weights(np.array(["A", "A", "A", "B"]))
    assert np.allclose(w, [1/3, 1/3, 1/3, 1.0])


def test_qc_filter_and_errors():
    df = table()
    df.loc[0, "n_steps"] = 3; df.loc[1, "fit_r2"] = 0.5
    kept, rej = qc_filter(df)
    assert rej == 2 and len(kept) == len(df) - 2
    with pytest.raises(ValueError, match="missing"):
        qc_filter(df.drop(columns=["inlier_frac"]))
    with pytest.raises(ValueError, match="subjects"):
        loso_evaluate(table(n_subj=3))


def test_v22_plausibility_qc():
    from das_height.features import gait_plausible
    assert gait_plausible(110, 0.7) and not gait_plausible(55, 1.4) and not gait_plausible(220, 0.35)
    assert not gait_plausible(float("nan"), 0.7)
    df = pd.DataFrame({"subject_id": ["a", "b"], "pass_id": ["a1", "b1"], "height_cm": [170.0, 170.0],
                       "n_steps": [20, 20], "fit_r2": [0.99, 0.99], "inlier_frac": [1.0, 1.0],
                       "step_length_m": [0.70, 1.40], "cadence_spm": [110.0, 55.0], "step_length_cv": [0.05, 0.05]})
    kept, rej = qc_filter(df)
    assert list(kept.pass_id) == ["a1"] and rej == 1
    kept21, rej21 = qc_filter(df, plausibility=False)
    assert len(kept21) == 2 and rej21 == 0


def test_v22_fill_ratio_qc():
    df = pd.DataFrame({"subject_id": ["a", "b"], "pass_id": ["a1", "b1"], "height_cm": [170.0, 170.0],
                       "n_steps": [20, 20], "fit_r2": [0.99, 0.99], "inlier_frac": [1.0, 1.0],
                       "step_length_m": [0.70, 0.70], "cadence_spm": [110.0, 110.0],
                       "step_length_cv": [0.05, 0.05], "fill_ratio": [0.98, 0.55]})
    kept, rej = qc_filter(df)
    assert list(kept.pass_id) == ["a1"] and rej == 1
