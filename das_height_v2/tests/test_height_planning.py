"""Phase B planning model (SIMULATED + ASSUMED)."""
import numpy as np
import pytest

from exploratory.height_planning.planning import (CONDITIONS, REPS_PER_CONDITION, loso_ols,
                                                  population, proven_above, simulate_passes, study)


def test_walk_ratio_model_is_exact_without_noise_or_style():
    rng = np.random.default_rng(0)
    H = population(50, rng)
    person, selfsel, cad, sl = simulate_passes(H, 0.0, rng)
    assert person.size == 50 * len(CONDITIONS) * REPS_PER_CONDITION
    ratio = sl / (cad * H[person])
    assert np.std(np.log(ratio)) < 0.01          # only pass noise (< 1 %) remains


def test_loso_does_not_use_the_held_out_person():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = loso_ols(x, y)
    assert p[-1] == pytest.approx(5.0)            # the outlier's own value never enters its prediction


def test_speed_compensation_recovers_height_when_style_is_small():
    r = study(400, 0.01, np.random.default_rng(1))
    assert r["r_B"] > 0.95 and r["rmse_B"] < 0.4 * r["rmse_baseline"]
    assert r["r_A"] < r["r_B"]                     # without cadence, speed differences hide height


def test_fisher_bound():
    assert proven_above(0.9, 30) and not proven_above(0.5, 12)
