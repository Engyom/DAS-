import numpy as np
from das_height.features import extract_pass_features


def walk(L=0.7, T=0.55, n=40, jitter_x=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = 1 + np.arange(n) * T
    x = 3 + np.arange(n) * L + jitter_x * rng.standard_normal(n)
    return t, x


def test_exact_walk():
    t, x = walk()
    f = extract_pass_features(t, x)
    assert abs(f["step_length_m"] - 0.7) < 1e-9
    assert abs(f["cadence_spm"] - 60 / 0.55) < 1e-6
    assert abs(f["speed_mps"] - 0.7 / 0.55) < 1e-9
    assert f["fit_r2"] > 0.999 and f["inlier_frac"] == 1.0


def test_missed_steps_do_not_change_step_length():
    t, x = walk()
    keep = np.ones(t.size, bool); keep[[5, 6, 20, 31]] = False
    f = extract_pass_features(t[keep], x[keep])
    assert abs(f["step_length_m"] - 0.7) < 1e-9


def test_false_detection_is_rejected_as_outlier():
    t, x = walk()
    x = x.copy(); x[10] += 8.0                      # one wrong position
    f = extract_pass_features(t, x)
    assert abs(f["step_length_m"] - 0.7) < 0.01
    assert f["inlier_frac"] < 1.0


def test_walking_backwards_gives_same_features():
    t, x = walk()
    f1 = extract_pass_features(t, x)
    f2 = extract_pass_features(t, x[::-1].copy())
    assert abs(f1["step_length_m"] - f2["step_length_m"]) < 1e-9


def test_too_few_steps_returns_nan():
    t, x = walk(n=4)
    f = extract_pass_features(t, x)
    assert np.isnan(f["step_length_m"]) and f["n_steps"] == 4


def test_false_detection_between_steps_does_not_change_gait_features():
    # v2.1 regression test: every gait feature uses the validated (inlier)
    # footsteps only. A false detection halfway in time between two real
    # steps, at a wrong position, used to halve one step period and inflate
    # step_length_cv.
    t, x = walk()
    t2 = np.insert(t, 11, (t[10] + t[11]) / 2)
    x2 = np.insert(x, 11, x[10] + 5.0)
    f0, f = extract_pass_features(t, x), extract_pass_features(t2, x2)
    assert f["n_detections"] == t.size + 1 and f["n_steps"] == t.size
    for k in ("step_length_m", "cadence_spm", "speed_mps", "step_length_direct_m"):
        assert abs(f[k] - f0[k]) < 1e-9, k
    assert f["step_length_cv"] < 1e-6
