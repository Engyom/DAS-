"""Footstep detection and localisation along the fibre.

Localisation uses the energy-weighted centre of each footstep's spatial
pattern, not the strongest channel. DAS senses axial strain only, so the
channel closest to a footstep beside the fibre can be weaker than its
neighbours (broadside dip); the pattern is roughly symmetric about the
footstep, so its centre is a better position estimate than its maximum.

v2 change: the detection threshold is set from a noise floor estimated
across channels (at any instant most channels are far from the walker)
instead of the 10th percentile of the envelope in time. In simulated tests
both gave the same results, including tightly cropped records and fast
cadences up to 190 steps/min; the new rule does not assume that 10% of a
recording is quiet, which real recordings may not satisfy.

v2.1 change: the band-pass uses mirror ("even") edge padding, 3 periods of
the lowest pass-band frequency long, instead of SciPy's default short odd
padding, which created false footsteps at the start and end of records
(see bandpass()).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import uniform_filter1d


def bandpass(x, fs, f1, f2, order=4, pad_periods=3.0):
    """Zero-phase Butterworth band-pass along axis 0 (time).

    Edge handling (v2.1). sosfiltfilt extends each end of the record before
    filtering. Its default extension is "odd" (point reflection about the
    first/last sample) and only ~27 samples long. On a noise-dominated edge
    the first sample is a random value, so odd reflection shifts the
    extension by 2*x[0]: a step at the record boundary. The band-pass step
    response then rings for ~0.9 s inside the data, and because the
    detection envelope is the maximum over all channels, the largest of
    these transients produced false footsteps at t ~ 0 and at the record
    end. Making the odd extension longer does not help (the step is at the
    boundary itself).

    "even" (mirror) reflection keeps the extension continuous with the data
    and has no level step. The extension is pad_periods periods of f1 long,
    so the filter's start-up transient decays inside the padding instead of
    inside the data (the default 27 samples is too short for f1 = 2 Hz).
    Records shorter than that are padded as much as possible (length - 1).
    Measured on the 180 simulated passes and on noise-only records; see
    CHANGES.md.
    """
    f2 = min(f2, 0.45 * fs)
    if not 0 < f1 < f2:
        raise ValueError(f"invalid band {f1}-{f2} Hz for fs={fs} Hz")
    sos = butter(order, [f1, f2], btype="band", fs=fs, output="sos")
    padlen = min(int(round(pad_periods * fs / f1)), x.shape[0] - 1)
    return sosfiltfilt(sos, x, axis=0, padtype="even", padlen=padlen)


def noise_floor(E):
    """Robust noise level of the energy map E (time x channel): the median
    over time of the median over channels."""
    return float(np.median(np.median(E, axis=1)))


def detect_footsteps(rec, band=(2.0, 60.0), win_s=0.1, min_sep_s=0.3,
                     thr_factor=8.0, loc_halfwidth_m=4.0, edge_m=2.0):
    """Return dict with footstep times [s], positions [m], energies and
    diagnostics.

    band            : pass band [Hz]. Measure the real footstep band at the
                      test site before fixing this (site dependent).
    win_s           : energy averaging window [s]
    min_sep_s       : minimum time between footsteps [s]
    thr_factor      : threshold = thr_factor x noise floor of the energy map
    loc_halfwidth_m : half-width of the window used for the energy centroid
    edge_m          : discard footsteps this close to either end of the section

    Data are used in their native units (strain or strain-rate): geometric
    features do not depend on units, and differentiating white noise
    amplifies high frequencies and can hide the footsteps.
    """
    if rec.data.ndim != 2 or rec.data.shape[0] < 3 * rec.fs * win_s:
        raise ValueError("record must be (time, channel) and longer than a few windows")
    x = bandpass(rec.data, rec.fs, band[0], band[1])
    E = uniform_filter1d(x ** 2, size=max(1, int(round(win_s * rec.fs))), axis=0)
    env = E.max(axis=1)
    floor = noise_floor(E)
    thr = thr_factor * floor
    pk, _ = find_peaks(env, height=thr, distance=max(1, int(round(min_sep_s * rec.fs))))

    pos_axis = rec.positions
    half = max(1, int(round(loc_halfwidth_m / rec.dx)))
    hw = max(1, int(round(0.08 * rec.fs)))
    times, positions, energies = [], [], []
    for p in pk:
        seg = x[max(0, p - hw): p + hw + 1]
        prof = (seg ** 2).sum(axis=0)                 # energy per channel
        ic = int(np.argmax(prof))
        lo, hi = max(0, ic - half), min(prof.size, ic + half + 1)
        w = np.clip(prof[lo:hi] - np.median(prof), 0, None)   # remove noise floor
        if w.sum() <= 0:
            continue
        times.append(p / rec.fs)
        positions.append(float((w * pos_axis[lo:hi]).sum() / w.sum()))
        energies.append(float(prof[ic]))
    times, positions, energies = map(np.asarray, (times, positions, energies))
    keep = (positions > pos_axis[0] + edge_m) & (positions < pos_axis[-1] - edge_m)
    return {"times": times[keep], "positions": positions[keep],
            "energies": energies[keep],
            "diagnostics": {"noise_floor": floor, "threshold": thr,
                            "n_peaks": int(pk.size), "n_kept": int(keep.sum())}}
