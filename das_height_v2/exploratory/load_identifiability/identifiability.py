"""T11: load / weight-range identifiability study (EXPLORATORY, SIMULATED).

Question: under which calibration conditions could footstep amplitude
distinguish body-weight ranges or a carried-load change?

Method: error propagation, NO classifier or regressor is trained.
In the generator (matlab/make_height_dataset.m, mirrored in tests/sim_py.py)
the footstep force is F = 1.15 * W * g * (1 + 0.05 N(0,1)) and the signal is
linear in F, so the footstep amplitude is proportional to weight BY
CONSTRUCTION. That proportionality is an assumption of the simulation, not
evidence. This study takes it as the best case and asks how large the other
amplitude changes (confounds) are compared with the weight effect:

    ln A = ln W + ln(lane term) + ln(coupling) + ln(speed) + ln(footwear) + noise

Separability of two weights:  d' = ln(W2/W1) / sigma_total.

Sources of each confound size:
  [Sim]        measured here from the generator equations (lane distance,
               channel grid, gauge length, step-to-step force, noise)
  [Assumption] swept over a stated range (coupling, speed, footwear); no
               value is backed by a citation yet (research plan, task T6).

Nothing here is evidence that DAS can estimate weight.
"""
import numpy as np
from scipy.stats import norm

from das_height.footsteps import detect_footsteps
from tests import sim_py

# ------------------------------------------------------------------ analytic
def dprime(w1, w2, sigma_total):
    """Separability of two weights given the total log-amplitude spread."""
    return abs(np.log(w2 / w1)) / sigma_total


def min_weight_ratio(sigma_total, d=2.0):
    """Smallest W2/W1 that reaches separability d."""
    return float(np.exp(d * sigma_total))


def n_classes(sigma_total, w_min=45.0, w_max=110.0, d=2.0):
    """Number of weight classes across [w_min, w_max] whose neighbours are
    separated by at least d (classes spaced evenly in log weight)."""
    return int(1 + np.floor(np.log(w_max / w_min) / (d * sigma_total)))


def adjacent_error(d):
    """Misclassification rate between two adjacent classes at separability d
    (equal Gaussian spreads, midpoint threshold)."""
    return float(norm.cdf(-d / 2.0))


def passes_needed(delta_ln, sigma_pass, alpha=0.05, power=0.8):
    """Passes per load level to detect a mean log-amplitude change delta_ln
    (two-sided two-sample test, normal approximation)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * (z * sigma_pass / delta_ln) ** 2))


def total_sigma(parts):
    return float(np.sqrt(sum(v ** 2 for v in parts.values())))


# ------------------------------------------------------------- simulation
def configure(dx=0.4, gl=0.8):
    """Set the generator mirror to channel spacing dx and gauge length gl.
    gl must be a whole multiple of dx so MATLAB and Python average the same
    number of channels (round(2.5) differs between them). With gl = n*dx the
    per-channel output noise equals the baseline level."""
    n = gl / dx
    if abs(n - round(n)) > 1e-9:
        raise ValueError("gauge length must be a whole multiple of channel spacing")
    sim_py.P["dx"] = dx
    return gl


def pass_amplitude(weight, lane, gl, rng, height=1.70):
    """Per-pass log amplitude: 0.5*ln(median footstep energy) from the
    pipeline's own detect_footsteps (peak-channel energy in +/-0.08 s)."""
    rec, _, _ = sim_py.simulate_pass(height, 0.415, lane, gl, rng, weight=weight)
    e = detect_footsteps(rec)["energies"]
    if e.size < 8:
        return np.nan, e
    return 0.5 * float(np.log(np.median(e))), 0.5 * np.log(e)


def single_footstep_peak(x_step, y_step, dx=0.4, gl=0.8, lfib=40.0):
    """Noise-free peak |strain| of ONE footstep at (x_step along the fibre,
    y_step lateral), using the generator's equations (lines in
    make_height_dataset.m: r, cos^2/sin^2 directivity, exp(-alpha r),
    gauge-length moving average). Returns the maximum over channels and
    time. Same physics, only an arbitrary footstep position."""
    S, G = sim_py.S, sim_py.G
    fs = sim_py.P["fs"]
    xc = np.arange(0, lfib + 1e-9, dx)
    w, dwv = sim_py.waveform(fs)
    rh2 = (xc - x_step) ** 2 + y_step ** 2
    rh2 = np.maximum(rh2, 1e-12)
    r = np.sqrt(rh2 + S["depth"] ** 2)
    cos2 = (xc - x_step) ** 2 / rh2
    sin2 = 1 - cos2
    A = S["G"] * G["aF"] * 70 * 9.81 * np.exp(-S["alpha"] * r)
    n0 = np.round(r / S["vs"] * fs).astype(int)               # arrival delay per channel
    X = np.zeros((xc.size, w.size + n0.max() + 1))
    for c in range(xc.size):
        X[c, n0[c]:n0[c] + w.size] = A[c] * (-dwv / S["vs"] * cos2[c] + w / r[c] * sin2[c])
    n = max(1, int(round(gl / dx)))
    if n > 1:
        k = np.ones(n) / n
        X = np.apply_along_axis(lambda col: np.convolve(col, k, "same"), 0, X)
    return float(np.abs(X).max())
