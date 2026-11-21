"""Python mirror of matlab/make_height_dataset.m, for testing the pipeline
without MATLAB. Same assumptions (see the MATLAB header). SIMULATED ONLY."""
import numpy as np
from das_height.schema import DASRecord

P = dict(fs=500.0, dx=0.25, Lfib=40.0, noise=0.05)
S = dict(depth=0.3, f0=25.0, zeta=0.15, alpha=0.15, G=1e-3, vs=200.0)
G = dict(kL_mean=0.415, kL_std=0.02, kL_pass_std=0.01,
         cad_range=(95.0, 125.0), aF=1.15, Tc=0.12)


def waveform(fs):
    t = np.arange(round(0.4 * fs)) / fs
    p = np.sin(np.pi * t / G["Tc"]) * (t <= G["Tc"])
    wd = 2 * np.pi * S["f0"] * np.sqrt(1 - S["zeta"] ** 2)
    h = np.exp(-S["zeta"] * 2 * np.pi * S["f0"] * t) * np.sin(wd * t)
    w = np.convolve(p, h)[:t.size]
    w /= np.abs(w).max()
    return w, np.gradient(w, 1 / fs)


def simulate_pass(H_m, k_subj, d, GL, rng, weight=70.0):
    fs, dx = P["fs"], P["dx"]
    xc = np.arange(0, P["Lfib"] + 1e-9, dx)
    L = k_subj * H_m * (1 + G["kL_pass_std"] * rng.standard_normal())
    cad = rng.uniform(*G["cad_range"])            # steps per minute
    T = 60.0 / cad
    xs = np.arange(3, P["Lfib"] - 3 + 1e-9, L)
    ts = 1.0 + np.arange(xs.size) * T + 0.01 * rng.standard_normal(xs.size)
    dk = d + 0.08 * (-1.0) ** np.arange(1, xs.size + 1)
    F = G["aF"] * weight * 9.81 * (1 + 0.05 * rng.standard_normal(xs.size))
    Nt = int(round((ts[-1] + 1.0) * fs))
    X = np.zeros((Nt, xc.size))
    w, dw = waveform(fs)
    for k in range(xs.size):
        rh2 = (xc - xs[k]) ** 2 + dk[k] ** 2           # horizontal distance^2
        r = np.sqrt(rh2 + S["depth"] ** 2)
        cos2 = (xc - xs[k]) ** 2 / rh2
        sin2 = 1 - cos2
        A = S["G"] * F[k] * np.exp(-S["alpha"] * r)
        n0 = np.round((ts[k] + r / S["vs"]) * fs).astype(int)
        for c in range(xc.size):
            e = min(n0[c] + w.size, Nt); m = e - n0[c]
            # axial strain of a radially outgoing wave:
            # du_r/dr * cos^2 + (u_r / r) * sin^2
            X[n0[c]:e, c] += A[c] * (-dw[:m] / S["vs"] * cos2[c] + w[:m] / r[c] * sin2[c])
    Aref = S["G"] * G["aF"] * 75 * 9.81
    X += P["noise"] * Aref * np.sqrt(GL / dx) * rng.standard_normal(X.shape)
    n = max(1, int(round(GL / dx)))
    if n > 1:
        k = np.ones(n) / n
        X = np.apply_along_axis(lambda row: np.convolve(row, k, "same"), 1, X)
    return DASRecord(X, fs, dx, GL, 0.0, "strain"), L, cad
