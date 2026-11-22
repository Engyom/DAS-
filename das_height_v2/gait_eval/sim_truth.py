"""Simulated passes WITH their ground truth (footstep times and positions).

tests/sim_py.simulate_pass returns the record, the step length and the
nominal cadence, but not the individual footsteps. It is not modified here.
Instead the random draws it makes are replayed from a copy of the same
generator state, which reproduces its footstep times and positions exactly:
    L   = k * H * (1 + kL_pass_std * N)     (draw 1)
    cad = U(cad_range)                      (draw 2)
    xs  = 3, 3+L, ... < Lfib-3
    ts  = 1 + i * 60/cad + 0.01 * N_i       (draws 3..)
SIMULATED ONLY: the "true" contact time is the generator's force onset.
"""
import copy
from contextlib import contextmanager

import numpy as np

from tests import sim_py


@contextmanager
def channel_spacing(dx):
    """Temporarily set the generator mirror's channel spacing."""
    old = sim_py.P["dx"]
    sim_py.P["dx"] = dx
    try:
        yield
    finally:
        sim_py.P["dx"] = old


def simulate_pass_with_truth(H_m, k_subj, lane_m, gl_m, rng, weight=70.0):
    """Returns (record, truth) where truth holds footstep times [s] and
    positions [m] plus the per-pass gait values implied by them."""
    replay = copy.deepcopy(rng)
    rec, L, cad = sim_py.simulate_pass(H_m, k_subj, lane_m, gl_m, rng, weight=weight)
    G, P = sim_py.G, sim_py.P
    L2 = k_subj * H_m * (1 + G["kL_pass_std"] * replay.standard_normal())
    cad2 = replay.uniform(*G["cad_range"])
    xs = np.arange(3, P["Lfib"] - 3 + 1e-9, L2)
    ts = 1.0 + np.arange(xs.size) * (60.0 / cad2) + 0.01 * replay.standard_normal(xs.size)
    if not (np.isclose(L, L2) and np.isclose(cad, cad2)):
        raise RuntimeError("replay of the generator draws diverged")
    T = 60.0 / cad2
    truth = {"times": ts, "positions": xs, "step_length_m": L2,
             "stride_length_m": 2 * L2, "cadence_spm": cad2, "speed_mps": L2 / T}
    return rec, truth


def reference_measurement(truth, rng, time_sd_s=0.005, pos_sd_m=0.01):
    """What a real ground-truth system would report: true contacts plus
    measurement error (video timing, footprint measurement). The SDs are
    ASSUMPTIONS until the dry-run uncertainty budget (research plan, sec. 4).
    Per-pass gait values are recomputed from the noisy contacts the same
    way a video/footprint analysis would."""
    t = truth["times"] + time_sd_s * rng.standard_normal(truth["times"].size)
    x = truth["positions"] + pos_sd_m * rng.standard_normal(truth["positions"].size)
    T = np.median(np.diff(t))
    step = np.median(np.abs(np.diff(x)))
    speed = abs(np.polyfit(t, x, 1)[0])
    return {"times": t, "positions": x, "step_length_m": float(step),
            "stride_length_m": float(2 * step), "cadence_spm": float(60 / T),
            "speed_mps": float(speed)}
