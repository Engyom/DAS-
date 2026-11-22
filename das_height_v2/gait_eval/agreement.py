"""Agreement and repeatability statistics for few participants (research plan 5.1).

Repeated-measures Bland-Altman
------------------------------
Several passes per participant, true value allowed to vary between passes
(Bland & Altman 2007, "Agreement between methods of measurement with
multiple observations per individual", J Biopharm Stat 17:571-582).
Differences d_ij = DAS - reference for pass j of participant i.
One-way ANOVA of d on participant gives
    s_w^2 = MS_within
    s_b^2 = (MS_between - MS_within) / D,  D = (n^2 - sum m_i^2) / ((k - 1) n)
    total SD = sqrt(s_b^2 + s_w^2)
    95 % limits of agreement = mean(d) +/- 1.96 * total SD
CIs for the limits use the MOVER method for this design (Zou 2013, Stat
Methods Med Res 22:630-642). With 3-5 participants they are wide; that is
the honest answer, not a defect.

Repeatability
-------------
Within-subject SD s_w from replicate passes of the same participant and
condition; repeatability coefficient RC = 2.77 * s_w (= 1.96 * sqrt(2) s_w),
the difference two repeat passes stay within 95 % of the time. CI from the
chi-square distribution with sum(m - 1) degrees of freedom.
"""
import numpy as np
from scipy import stats

Z95 = 1.959963984540054


def _groups(values, groups):
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    labels = list(dict.fromkeys(g.tolist()))
    return [v[g == u] for u in labels], labels


def rm_bland_altman(das, ref, subject, alpha=0.05):
    """Bland-Altman for repeated measures. Returns a dict with bias, total
    SD, LoA and their CIs, variance components and per-subject bias."""
    d = np.asarray(das, float) - np.asarray(ref, float)
    parts, labels = _groups(d, subject)
    parts = [p for p in parts if p.size]
    k = len(parts)
    m = np.array([p.size for p in parts], float)
    n = m.sum()
    if k < 2 or n - k < 1:
        raise ValueError("need at least 2 participants and one repeat")
    allv = np.concatenate(parts)
    bias = float(allv.mean())
    means = np.array([p.mean() for p in parts])
    ss_b = float((m * (means - bias) ** 2).sum())
    ss_w = float(sum(((p - p.mean()) ** 2).sum() for p in parts))
    df_b, df_w = k - 1, int(n - k)
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    D = (n ** 2 - (m ** 2).sum()) / ((k - 1) * n)
    s2_b = max(0.0, (ms_b - ms_w) / D)
    s2_w = ms_w
    sd = float(np.sqrt(s2_b + s2_w))

    # bias CI: Var(mean) = s_b^2 sum(m^2)/n^2 + s_w^2/n, t with k-1 df
    se_bias = np.sqrt(s2_b * (m ** 2).sum() / n ** 2 + s2_w / n)
    tq = stats.t.ppf(1 - alpha / 2, df_b)
    lb, ub = bias - tq * se_bias, bias + tq * se_bias

    # MOVER CI for total variance = MS_b / D + (1 - 1/D) MS_w
    c = np.array([1 / D, 1 - 1 / D]); ms = np.array([ms_b, ms_w]); df = np.array([df_b, df_w])
    est_var = float((c * ms).sum())
    lo_f = 1 - df / stats.chi2.ppf(1 - alpha / 2, df)
    hi_f = df / stats.chi2.ppf(alpha / 2, df) - 1
    var_lo = est_var - np.sqrt(((c * ms * lo_f) ** 2).sum())
    var_hi = est_var + np.sqrt(((c * ms * hi_f) ** 2).sum())
    sd_est = np.sqrt(est_var)
    sd_lo, sd_hi = np.sqrt(max(var_lo, 0.0)), np.sqrt(var_hi)

    def loa_ci(sign):
        loa = bias + sign * Z95 * sd_est
        if sign > 0:
            lo = loa - np.sqrt((bias - lb) ** 2 + (Z95 * (sd_est - sd_lo)) ** 2)
            hi = loa + np.sqrt((ub - bias) ** 2 + (Z95 * (sd_hi - sd_est)) ** 2)
        else:
            lo = loa - np.sqrt((bias - lb) ** 2 + (Z95 * (sd_hi - sd_est)) ** 2)
            hi = loa + np.sqrt((ub - bias) ** 2 + (Z95 * (sd_est - sd_lo)) ** 2)
        return float(loa), float(lo), float(hi)

    up, low = loa_ci(+1), loa_ci(-1)
    return {
        "n_subjects": k, "n_pairs": int(n),
        "bias": bias, "bias_ci": (float(lb), float(ub)),
        "sd_total": sd, "sd_between": float(np.sqrt(s2_b)), "sd_within": float(np.sqrt(s2_w)),
        "loa_low": low[0], "loa_low_ci": low[1:],
        "loa_high": up[0], "loa_high_ci": up[1:],
        "per_subject_bias": dict(zip(labels, means.tolist())),
    }


def proportional_bias(das, ref, subject):
    """Slope of difference on mean, fitted on participant means (avoids
    treating passes as independent). Returns slope, its CI and p-value;
    NaN when fewer than 3 participants."""
    d = np.asarray(das, float) - np.asarray(ref, float)
    a = (np.asarray(das, float) + np.asarray(ref, float)) / 2
    dp, _ = _groups(d, subject)
    ap, _ = _groups(a, subject)
    x = np.array([p.mean() for p in ap]); y = np.array([p.mean() for p in dp])
    if x.size < 3 or np.ptp(x) == 0:
        return {"slope": np.nan, "slope_ci": (np.nan, np.nan), "p": np.nan, "n_subjects": int(x.size)}
    r = stats.linregress(x, y)
    tq = stats.t.ppf(0.975, x.size - 2)
    return {"slope": float(r.slope), "slope_ci": (float(r.slope - tq * r.stderr), float(r.slope + tq * r.stderr)),
            "p": float(r.pvalue), "n_subjects": int(x.size)}


def repeatability(values, cell, alpha=0.05):
    """Within-subject SD and repeatability coefficient from replicate passes.
    `cell` identifies one participant-condition combination (e.g.
    'P1|lane1.0|110spm'); cells with a single pass contribute nothing."""
    parts, _ = _groups(values, cell)
    parts = [p for p in parts if p.size >= 2]
    df = int(sum(p.size - 1 for p in parts))
    if df < 1:
        raise ValueError("need replicate passes")
    ss = float(sum(((p - p.mean()) ** 2).sum() for p in parts))
    sw = np.sqrt(ss / df)
    lo = np.sqrt(ss / stats.chi2.ppf(1 - alpha / 2, df))
    hi = np.sqrt(ss / stats.chi2.ppf(alpha / 2, df))
    k = Z95 * np.sqrt(2)
    return {"sd_within": float(sw), "sd_within_ci": (float(lo), float(hi)),
            "rc": float(k * sw), "rc_ci": (float(k * lo), float(k * hi)),
            "df": df, "n_cells": len(parts)}
