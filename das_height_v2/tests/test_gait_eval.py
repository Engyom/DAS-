"""T9 analysis tooling: matching, agreement, repeatability, simulated truth."""
import numpy as np
import pytest

from gait_eval.agreement import proportional_bias, repeatability, rm_bland_altman
from gait_eval.matching import detection_metrics, estimate_offset, match_footsteps
from gait_eval.sim_truth import channel_spacing, simulate_pass_with_truth
from tests import sim_py


def test_matching_one_to_one_nearest_first():
    gt = np.array([1.0, 1.5, 2.0])
    det = np.array([1.01, 1.02, 1.49, 3.0])        # two detections near the first contact
    m = match_footsteps(det, gt, tol_s=0.05)
    assert m["pairs"].tolist() == [[0, 0], [2, 1]]  # nearest wins, second is a false detection
    assert set(m["unmatched_det"].tolist()) == {1, 3}
    assert m["unmatched_gt"].tolist() == [2]
    r = detection_metrics(det, gt, tol_s=0.05)
    assert r["recall"] == pytest.approx(2 / 3) and r["precision"] == pytest.approx(0.5)


def test_offset_is_estimated_then_removed():
    rng = np.random.default_rng(0)
    gt = 1.0 + 0.5 * np.arange(40)
    det = gt + 0.06 + 0.005 * rng.standard_normal(gt.size)
    off, n = estimate_offset(det, gt)
    assert off == pytest.approx(0.06, abs=0.003) and n == gt.size
    assert detection_metrics(det, gt, tol_s=0.05)["recall"] < 0.5      # raw: lag breaks matching
    assert detection_metrics(det, gt, off, tol_s=0.05)["recall"] == 1.0


def test_rm_bland_altman_recovers_variance_components():
    rng = np.random.default_rng(1)
    k, m, sb, sw, bias = 3000, 3, 0.02, 0.01, 0.005
    subj = np.repeat(np.arange(k), m)
    ref = rng.uniform(1.2, 1.6, k * m)
    das = ref + bias + np.repeat(sb * rng.standard_normal(k), m) + sw * rng.standard_normal(k * m)
    ba = rm_bland_altman(das, ref, subj)
    assert ba["bias"] == pytest.approx(bias, abs=0.002)
    assert ba["sd_between"] == pytest.approx(sb, rel=0.1)
    assert ba["sd_within"] == pytest.approx(sw, rel=0.05)
    sd = np.hypot(sb, sw)
    assert ba["loa_high"] == pytest.approx(ba["bias"] + 1.96 * ba["sd_total"], rel=1e-4)
    assert ba["loa_high"] == pytest.approx(bias + 1.96 * sd, abs=0.002)
    assert ba["loa_high_ci"][0] < ba["loa_high"] < ba["loa_high_ci"][1]
    assert ba["loa_low_ci"][0] < ba["loa_low"] < ba["loa_low_ci"][1]


def test_few_subjects_give_wide_but_valid_cis():
    rng = np.random.default_rng(2)
    subj = np.repeat(["A", "B", "C", "D"], 9)
    ref = rng.uniform(1.2, 1.6, subj.size)
    das = ref + 0.01 * rng.standard_normal(subj.size)
    ba = rm_bland_altman(das, ref, subj)
    w = ba["loa_high_ci"][1] - ba["loa_high_ci"][0]
    assert w > 0.25 * (ba["loa_high"] - ba["loa_low"])                  # imprecise, as it should be
    assert np.isnan(proportional_bias(das[:18], ref[:18], subj[:18])["slope"])   # 2 subjects: not estimated


def test_repeatability_coefficient():
    rng = np.random.default_rng(3)
    cells = np.repeat(np.arange(300), 3)
    v = np.repeat(rng.uniform(1, 2, 300), 3) + 0.02 * rng.standard_normal(cells.size)
    r = repeatability(v, cells)
    assert r["sd_within"] == pytest.approx(0.02, rel=0.06)
    assert r["rc"] == pytest.approx(2.77 * r["sd_within"], rel=1e-3)
    assert r["df"] == 600


def test_sim_truth_replay_matches_generator_and_restores_spacing():
    old = sim_py.P["dx"]
    with channel_spacing(0.4):
        rec, truth = simulate_pass_with_truth(1.70, 0.415, 1.0, 0.8, np.random.default_rng(5))
        assert rec.dx == 0.4
    assert sim_py.P["dx"] == old
    _, L, cad = sim_py.simulate_pass(1.70, 0.415, 1.0, 0.8, np.random.default_rng(5))
    assert truth["step_length_m"] == pytest.approx(L) and truth["cadence_spm"] == pytest.approx(cad)
    assert np.all(np.diff(truth["positions"]) == pytest.approx(L))


# ---------------------------------------------------------------- T10
from gait_eval.sensitivity import detection_snr_db, gauge_for, generator, run_setting


def test_gauge_is_whole_multiple_of_spacing():
    assert [gauge_for(d) for d in (0.25, 0.4, 0.5, 1.0, 2.0)] == [1.0, 0.8, 1.0, 1.0, 2.0]


def test_generator_settings_are_restored():
    old = (sim_py.P["dx"], sim_py.P["noise"], sim_py.G["cad_range"])
    with generator(2.0, noise_mult=4.0, cadence=130.0):
        assert sim_py.P["dx"] == 2.0 and sim_py.G["cad_range"] == (130.0, 130.0)
    assert (sim_py.P["dx"], sim_py.P["noise"], sim_py.G["cad_range"]) == old


def test_detection_snr_falls_with_noise_and_setting_runs():
    rng = np.random.default_rng(9)
    snr = []
    for nm in (1.0, 4.0):
        with generator(0.4, nm):
            rec, truth = simulate_pass_with_truth(1.7, 0.415, 1.0, 0.8, rng)
        snr.append(detection_snr_db(rec, truth["times"] + 0.055))
    assert snr[0] - snr[1] == pytest.approx(20 * np.log10(4), abs=2.0)   # energy SNR: -12 dB for 4x noise
    rows, det = run_setting(2, np.random.default_rng(1), 0.4, 0.8, 1.0)
    assert len(rows) == 2 and det["recall"] > 0.9
