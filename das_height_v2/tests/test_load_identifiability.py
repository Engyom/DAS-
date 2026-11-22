"""Tests for the T11 exploratory identifiability study (SIMULATED, no model)."""
import numpy as np
import pytest

from exploratory.load_identifiability import identifiability as I
from tests import sim_py


@pytest.fixture(autouse=True)
def restore_dx():
    dx = sim_py.P["dx"]
    yield
    sim_py.P["dx"] = dx


def test_dprime_and_min_ratio_are_consistent():
    s = 0.2
    r = I.min_weight_ratio(s, d=2.0)
    assert np.isclose(I.dprime(60.0, 60.0 * r, s), 2.0)


def test_n_classes():
    # ln(110/45) = 0.894: at sigma 0.5 neighbours need ln-ratio 1.0 -> one class
    assert I.n_classes(0.5) == 1
    assert I.n_classes(0.2) == 3          # 0.894 / 0.4 = 2.2 -> 1 + 2
    assert I.n_classes(0.01) > 40


def test_adjacent_error_and_sample_size():
    assert np.isclose(I.adjacent_error(0.0), 0.5)
    assert I.adjacent_error(4.0) < 0.03
    assert I.passes_needed(np.log(1.2), 0.10) < I.passes_needed(np.log(1.1), 0.10)


def test_gauge_length_must_be_multiple_of_spacing():
    with pytest.raises(ValueError):
        I.configure(dx=0.4, gl=1.0)
    assert I.configure(dx=0.4, gl=0.8) == 0.8


def test_single_footstep_amplitude_falls_with_lateral_distance():
    near = I.single_footstep_peak(20.0, 0.5, dx=0.4, gl=0.8)
    far = I.single_footstep_peak(20.0, 2.0, dx=0.4, gl=0.8)
    assert far < near


def test_pass_amplitude_scales_with_weight_as_built_into_the_generator():
    # This only confirms the generator's own assumption (amplitude ~ weight);
    # it is not evidence that DAS measures weight.
    I.configure(dx=0.4, gl=0.8)
    a60, _ = I.pass_amplitude(60.0, 1.0, 0.8, np.random.default_rng(1))
    a90, _ = I.pass_amplitude(90.0, 1.0, 0.8, np.random.default_rng(1))
    assert abs((a90 - a60) - np.log(90 / 60)) < 0.05
