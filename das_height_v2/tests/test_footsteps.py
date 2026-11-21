import numpy as np
import pytest
from das_height.schema import DASRecord
from das_height.footsteps import detect_footsteps
from das_height.features import extract_pass_features
from tests.sim_py import simulate_pass


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0])
def test_step_length_recovered_at_each_lane(d):
    rng = np.random.default_rng(int(10 * d))
    rec, L, cad = simulate_pass(1.65, 0.415, d, 1.0, rng)
    fs = detect_footsteps(rec)
    f = extract_pass_features(fs["times"], fs["positions"])
    assert f["n_steps"] >= 30
    assert abs(f["step_length_m"] - L) < 0.02            # 2 cm
    assert abs(f["cadence_spm"] - cad) < 2.0


def test_tightly_cropped_record_still_detected():
    # A record cropped to the walk itself has little quiet time; the
    # threshold must not depend on quiet time being present.
    rng = np.random.default_rng(3)
    rec, L, cad = simulate_pass(1.70, 0.415, 1.0, 1.0, rng)
    i0, i1 = int(1.2 * rec.fs), rec.data.shape[0] - int(1.2 * rec.fs)
    cropped = DASRecord(rec.data[i0:i1], rec.fs, rec.dx, rec.gauge_length)
    fs = detect_footsteps(cropped)
    f = extract_pass_features(fs["times"], fs["positions"])
    assert f["n_steps"] >= 30
    assert abs(f["step_length_m"] - L) < 0.02


def test_noise_only_gives_almost_no_detections():
    rng = np.random.default_rng(4)
    rec, _, _ = simulate_pass(1.65, 0.415, 1.0, 1.0, rng)
    noise = DASRecord(rng.standard_normal(rec.data.shape) * rec.data.std(),
                      rec.fs, rec.dx, rec.gauge_length)
    fs = detect_footsteps(noise)
    assert len(fs["times"]) <= 2


def test_rejects_wrong_orientation():
    with pytest.raises(ValueError):
        detect_footsteps(DASRecord(np.zeros((10, 500)), 500.0, 0.25, 1.0))


def test_no_false_detections_at_record_edges():
    # v2.1 regression test. With SciPy's default short odd edge padding the
    # band-pass step response produced false footsteps at t ~ 0 and at the
    # record end on noise-only data (max over many channels). Mirror padding
    # of 3 periods of the low band edge removes them.
    rng = np.random.default_rng(7)
    n_edge = 0
    for _ in range(5):
        x = rng.standard_normal((6000, 161))
        d = detect_footsteps(DASRecord(x, 500.0, 0.25, 1.0))
        t = d["times"]
        n_edge += int(((t < 0.2) | (t > 12.0 - 0.2)).sum())
    assert n_edge == 0


def test_footstep_near_record_start_is_kept():
    # Edge handling must not remove a real footstep close to the edge:
    # crop a simulated pass 0.2 s before its first footstep.
    rng = np.random.default_rng(5)
    rec, L, cad = simulate_pass(1.65, 0.415, 1.0, 1.0, rng)
    full = detect_footsteps(rec)["times"]
    first = full[full > 0.5][0]                # simulated walks start at ~1 s
    i0 = int(round((first - 0.25) * rec.fs))
    cropped = DASRecord(rec.data[i0:], rec.fs, rec.dx, rec.gauge_length)
    t = detect_footsteps(cropped)["times"] + i0 / rec.fs
    assert np.min(np.abs(t - first)) < 0.04
