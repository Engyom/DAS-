"""Footstep matching: DAS detections to ground-truth contacts (research plan 5.2).

Procedure (fixed before data collection):
1. Estimate the systematic DAS-minus-ground-truth time offset as the median
   of nearest-neighbour differences (on dry-run data in the real study).
2. Subtract that offset from the DAS times.
3. Match one-to-one, nearest pairs first, within +/- tol (default 50 ms):
   each ground-truth contact gets at most one detection and vice versa.
4. Report recall, precision and false detections, and repeat for the
   tolerance sensitivity set (25, 50, 100, 150 ms).

The offset is reported as timing bias; it is never absorbed silently.
"""
import numpy as np

TOLERANCES_S = (0.025, 0.050, 0.100, 0.150)


def estimate_offset(det_times, gt_times, max_abs_s=0.25):
    """Median of (detection - nearest ground-truth contact), using only pairs
    closer than max_abs_s (so gross false detections do not pull it).
    Returns (offset_s, n_pairs_used)."""
    det = np.asarray(det_times, float)
    gt = np.sort(np.asarray(gt_times, float))
    if det.size == 0 or gt.size == 0:
        return np.nan, 0
    idx = np.clip(np.searchsorted(gt, det), 1, gt.size - 1) if gt.size > 1 else np.zeros(det.size, int)
    if gt.size > 1:
        left, right = gt[idx - 1], gt[idx]
        nearest = np.where(np.abs(det - left) <= np.abs(det - right), left, right)
    else:
        nearest = np.full(det.size, gt[0])
    d = det - nearest
    d = d[np.abs(d) <= max_abs_s]
    return (float(np.median(d)) if d.size else np.nan), int(d.size)


def match_footsteps(det_times, gt_times, offset_s=0.0, tol_s=0.050):
    """One-to-one matching, nearest pairs first, within +/- tol_s after
    removing offset_s from the detection times.

    Returns dict:
      pairs     : int array (n_matched, 2) of (det_index, gt_index)
      dt        : corrected detection time minus contact time, per pair [s]
      unmatched_det, unmatched_gt : index arrays
    """
    det = np.asarray(det_times, float) - offset_s
    gt = np.asarray(gt_times, float)
    if det.size == 0 or gt.size == 0:
        return {"pairs": np.zeros((0, 2), int), "dt": np.zeros(0),
                "unmatched_det": np.arange(det.size), "unmatched_gt": np.arange(gt.size)}
    diff = det[:, None] - gt[None, :]
    i, j = np.nonzero(np.abs(diff) <= tol_s)
    order = np.argsort(np.abs(diff[i, j]), kind="stable")
    used_d, used_g, pairs = set(), set(), []
    for k in order:
        a, b = int(i[k]), int(j[k])
        if a in used_d or b in used_g:
            continue
        used_d.add(a); used_g.add(b); pairs.append((a, b))
    pairs = np.array(sorted(pairs), int).reshape(-1, 2)
    dt = det[pairs[:, 0]] - gt[pairs[:, 1]] if pairs.size else np.zeros(0)
    return {"pairs": pairs, "dt": dt,
            "unmatched_det": np.setdiff1d(np.arange(det.size), pairs[:, 0]),
            "unmatched_gt": np.setdiff1d(np.arange(gt.size), pairs[:, 1])}


def detection_metrics(det_times, gt_times, offset_s=0.0, tol_s=0.050):
    """Recall, precision, false detections and residual timing for one pass."""
    m = match_footsteps(det_times, gt_times, offset_s, tol_s)
    n_det, n_gt, n_tp = len(det_times), len(gt_times), len(m["pairs"])
    return {
        "tol_ms": 1000 * tol_s,
        "n_gt": n_gt, "n_det": n_det, "n_matched": n_tp,
        "n_false": n_det - n_tp, "n_missed": n_gt - n_tp,
        "recall": n_tp / n_gt if n_gt else np.nan,
        "precision": n_tp / n_det if n_det else np.nan,
        "timing_resid_mean_ms": 1000 * float(m["dt"].mean()) if n_tp else np.nan,
        "timing_resid_sd_ms": 1000 * float(m["dt"].std(ddof=1)) if n_tp > 1 else np.nan,
    }


def tolerance_sweep(det_times, gt_times, offset_s=0.0, tolerances=TOLERANCES_S):
    """detection_metrics at every tolerance of the sensitivity set."""
    return [detection_metrics(det_times, gt_times, offset_s, t) for t in tolerances]


def match_positions(det_pos, gt_pos, pairs):
    """Localisation error (DAS minus true position) for matched pairs [m]."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros(0)
    return np.asarray(det_pos, float)[pairs[:, 0]] - np.asarray(gt_pos, float)[pairs[:, 1]]
