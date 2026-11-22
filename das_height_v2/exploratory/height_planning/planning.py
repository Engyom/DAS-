"""Phase B planning model for height from gait (SIMULATED + ASSUMED).

Question: how much stature information can a speed-compensated gait model
recover, and how many participants does Phase B need to show it? This is a
planning tool, not evidence. No DAS signal is simulated: the DAS enters only
through its measured step-length error (T10).

Gait model (walk-ratio form): step length / cadence is roughly constant within
a person over a wide speed range (Sekiya et al. 1996; Sekiya & Nagasaki 1998)
and, standardised by stature, similar across populations (Adachi et al.):

    step_length = K * height * cadence * exp(u_i) * exp(e)

u_i ~ N(0, sigma_u)  person-specific gait style: the unknown biological limit
e                    pass noise: 1.5 % step-to-step CV left after speed
                     (PMC3756042) averaged over N_STEPS, plus DAS error (T10)

ASSUMPTIONS (labelled, to be replaced by Phase A/B measurements):
  population 50 % men N(175, 7) cm, 50 % women N(162, 6.5) cm;
  self-selected cadence N(112, CV) between people, CV 8 % (or 12 %);
  40 steady steps per pass; DAS step-length error 0.5 %.
Not modelled: walk-ratio drift at very slow speed, footwear, terrain, load.

Models, leave-one-subject-out OLS on log features:
  A  height ~ ln(step length)                  one self-selected pass
  B  height ~ ln(step length) + ln(cadence)    one self-selected pass (field case)
  C  height ~ person mean of ln(SL / cadence)  all conditions (research curve)
"""
import numpy as np

K = 0.415 / 112.0
CONDITIONS = (None, 90.0, 100.0, 110.0, 120.0, 130.0)   # None = self-selected
REPS_PER_CONDITION = 3
N_STEPS = 40
STEP_NOISE_CV = 0.015
DAS_ERROR_CV = 0.005


def population(n, rng):
    male = rng.random(n) < 0.5
    return np.where(male, rng.normal(175.0, 7.0, n), rng.normal(162.0, 6.5, n))


def simulate_passes(H, sigma_u, rng, cadence_cv=0.08):
    """Return arrays (person, is_self_selected, cadence, step_length) for every pass."""
    n = H.size
    u = rng.normal(0.0, sigma_u, n)
    c_self = rng.normal(112.0, 112.0 * cadence_cv, n)
    pass_sd = np.hypot(STEP_NOISE_CV / np.sqrt(N_STEPS), DAS_ERROR_CV)
    person, selfsel, cad = [], [], []
    for c in CONDITIONS:
        for _ in range(REPS_PER_CONDITION):
            person.append(np.arange(n))
            selfsel.append(np.full(n, c is None))
            cad.append(c_self if c is None else np.full(n, c))
    person, selfsel, cad = map(np.concatenate, (person, selfsel, cad))
    sl = K * H[person] * cad * np.exp(u[person] + rng.normal(0.0, pass_sd, person.size))
    return person, selfsel, cad, sl


def loso_ols(X, y):
    """Leave-one-subject-out ordinary least squares (one row per person)."""
    X = np.atleast_2d(np.asarray(X, float).T).T
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        A = np.c_[np.ones(m.sum()), X[m]]
        b = np.linalg.lstsq(A, y[m], rcond=None)[0]
        pred[i] = np.r_[1.0, X[i]] @ b
    return pred


def study(n, sigma_u, rng, cadence_cv=0.08):
    """One simulated Phase B study. Returns per-model RMSE and r on held-out people."""
    H = population(n, rng)
    person, selfsel, cad, sl = simulate_passes(H, sigma_u, rng, cadence_cv)
    first = np.array([np.flatnonzero((person == i) & selfsel)[0] for i in range(n)])
    lsl, lc = np.log(sl[first]), np.log(cad[first])
    curve = np.array([np.mean(np.log(sl[person == i] / cad[person == i])) for i in range(n)])
    preds = {"A": loso_ols(lsl, H), "B": loso_ols(np.c_[lsl, lc], H), "C": loso_ols(curve, H)}
    base = (H.sum() - H) / (n - 1)
    out = {"rmse_baseline": float(np.sqrt(np.mean((base - H) ** 2)))}
    for k, p in preds.items():
        out[f"rmse_{k}"] = float(np.sqrt(np.mean((p - H) ** 2)))
        out[f"r_{k}"] = float(np.corrcoef(p, H)[0, 1])
    return out


def proven_above(r, n, r0=0.45, alpha=0.05):
    """True if the one-sided (1 - alpha) lower Fisher-z bound of r exceeds r0."""
    from scipy.stats import norm
    return bool(np.tanh(np.arctanh(r) - norm.ppf(1 - alpha) / np.sqrt(n - 3)) > r0)
