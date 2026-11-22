# gait_eval/ — Phase A analysis tooling (task T9)

Tools to compare DAS gait measurements with a ground-truth reference (video
contacts, heel prints), written and fixed **before** any real data exist
(research plan, section 5). Imports `das_height`; never changes it.

| File | Contents |
|---|---|
| `matching.py` | DAS–reference offset (median of nearest differences), one-to-one nearest-first matching within ±50 ms, recall / precision / false detections, tolerance sweep 25–150 ms, localisation error |
| `agreement.py` | Repeated-measures Bland–Altman (Bland & Altman 2007, true value varying), MOVER CIs for the limits (Zou 2013), bias CI, proportional bias on participant means, within-subject SD and repeatability coefficient (2.77 × s_w) with χ² CIs |
| `sim_truth.py` | Simulated passes **with** their true footstep times and positions (replays the generator's random draws; `tests/sim_py.py` is not modified); reference measurement with assumed error |
| `run_sim_demo.py` | The whole Phase A chain on simulated passes: dry run → offset → matching → detection → timing / localisation → Bland–Altman → repeatability |
| `results/` | Output of the demo (CSV, `summary.json`, `bland_altman_sim.png`) |

Run: `python -m gait_eval.run_sim_demo --out gait_eval/results`
Tests: `tests/test_gait_eval.py` (6 tests).

## Tooling check on simulated passes — SIMULATED, not accuracy

Design as in Phase A: 4 participants × lanes 0.5 / 1 / 2 m × 3 passes, channel
spacing 0.4 m, gauge length 0.8 m, seed 11. Reference error **assumed**:
5 ms video timing SD, 1 cm footprint SD (to be replaced by the dry-run
uncertainty budget). The generator is clean (white noise, ideal gauge), so
the agreement below is a best case and says nothing about real data.

**Detection** (1 719 contacts): offset from the dry run 57 ms (DAS later than
contact, the expected energy-window lag).

| Tolerance | 25 ms | 50 ms | 100 ms | 150 ms |
|---|---|---|---|---|
| Recall | 84.0 % | 99.9 % | 100 % | 100 % |
| Precision | 83.4 % | 99.1 % | 99.2 % | 99.2 % |

**Agreement, DAS vs reference** (95 % LoA; in brackets the far CI bounds):

| Quantity | Lanes | LoA | Far CI bound | Provisional threshold | Point estimate | Far CI bound |
|---|---|---|---|---|---|---|
| Stride length | ≤ 1 m | −1.6 / +1.3 cm | ±3.3 cm | ±5 cm | within | within |
| Stride length | 2 m | −2.6 / +2.0 cm | ±6.1 cm | ±8 cm | within | within |
| Cadence | all | −1.1 / +1.3 spm | ±2.6 spm | ±2 spm | within | **outside** |
| Cadence | 2 m | −1.2 / +1.6 spm | ±4.1 spm | ±2 spm | within | **outside** |
| Speed | all | −0.3 / +0.1 cm/s | ±0.4 cm/s | ±5 cm/s | within | within |

## What the check found (to act on before Phase A)
1. **±50 ms is the right tolerance; ±25 ms is not.** At 25 ms recall falls to
   84 % although the detections are correct: the residual timing SD is
   9–16 ms. Keep ±50 ms and report the sweep, as planned.
2. **The DAS–reference offset depends on lane distance.** With one pooled
   offset the residual timing bias is −1, +4 and +13 ms at 0.5, 1 and 2 m
   (wave travel time grows with distance). Estimate the offset per lane in
   the dry run.
3. **Few participants make the CI-based decision hard for cadence.** The
   point-estimate LoA meets ±2 spm, but with 4 participants × 3 passes the
   far CI bound reaches 2.6 spm (4.1 at 2 m). Decision D3 should say whether
   acceptance uses the point estimate or the CI bound; if the CI bound, Phase A
   needs more passes per lane or a wider cadence threshold.
4. **Localisation bias of +0.21 m at every lane comes from the simulation,
   not the pipeline.** The Python mirror averages over an even number of
   channels (0.8 / 0.4 = 2) with `np.convolve(..., "same")`, which shifts the
   field by half a channel (+0.2 m). MATLAB's `conv2(..., 'same')` in
   `make_height_dataset.m` centres an even kernel the other way (documented
   behaviour, not yet run here), so the two generators would disagree by one
   channel in absolute position. Stride, cadence and speed are unaffected
   (they use differences). Neither generator file is changed; with odd
   gl/dx the shift vanishes.
5. Repeatability: DAS within-subject SD of stride 1.75 cm vs 1.66 cm for the
   true values, so in the simulation almost all pass-to-pass spread is the
   walker's, not the sensor's. Cadence and speed SDs are large only because
   the generator draws a new cadence for every pass (95–125 spm); with a
   metronome in Phase A they will be much smaller.

---

# T10 — gait sensitivity study (SIMULATED)

`sensitivity.py` sweeps one factor at a time around a baseline (spacing 0.4 m,
gauge length 0.8 m, lane 1 m, default noise, cadence 95–125 steps/min) plus a
noise × lane grid; 12 passes per setting, seed 7. Errors are DAS vs the **true**
simulated footsteps (sensor + pipeline only). Gauge length is not swept
(research plan 6.1); for the spacing sweep it is the smallest whole multiple of
the spacing ≥ 0.8 m, so per-channel noise is the same at every spacing.
Output: `results_sensitivity/` (`sensitivity.csv`, `sensitivity.png`).
Run: `python -m gait_eval.sensitivity` (about 3 min).

All results are conditional on the generator (white noise, ideal gauge, one
homogeneous ground). They show where the method breaks in this model, not how
well it works in the field.

## Where it works
| Factor | Result |
|---|---|
| Channel spacing 0.25 → 2 m | Stride error SD ≤ 1.7 cm at every spacing and lane 0.5–2 m (indirect stride = speed × step period averages over many steps). Localisation SD grows: 1.7–6.8 cm at 0.4 m, 20–39 cm at 2 m |
| Lane distance 0.5 → 3 m (default noise) | Detection SNR falls from 27.7 to 12.8 dB; QC 100 %, recall 100 %, stride SD ≤ 1.2 cm throughout |
| Cadence 90 → 140 steps/min | Recall 100 %, precision ≥ 99.2 %, stride SD ≤ 0.7 cm |
| Noise ×0.5 → ×16 (lane 1 m) | Works down to ×4 (SNR 12.2 dB); at ×8 (9.3 dB) recall falls to 6 % and no pass passes QC. The cliff sits at the detection threshold (8 × noise floor = 9.0 dB) |

Operating rule for Phase 0 (task T16): measure the detection SNR with
`detection_snr_db` on real taps and walks; in this model the method needs
≳ 12 dB, and the grid shows the lane distance at which that is reached for
each noise level.

## Two failure modes of the v2.1 pipeline (found here, not fixed)
1. **Too much signal: band-pass ringing becomes footsteps.** At noise ×0.5
   (SNR 29 dB), 3 of 12 passes doubled their cadence (+94 to +96 steps/min)
   and stride error was −74 to −80 cm, yet they passed QC. The zero-phase
   2–60 Hz Butterworth band-pass makes envelope side-lobes 0.28 s before and
   0.41 s after each footstep, about 24 dB below it (none in the raw
   envelope). Once a footstep is ~33 dB above the noise floor, the side-lobes
   cross the 9 dB threshold; they are more than the 0.3 s minimum separation
   away, so they are kept, and they lie on the walker's track, so the robust
   fit keeps them too. The same effect already lowers precision to 94–99.5 %
   at lane 0.5 m with default noise (vs 100 % at 1 m). Real recordings close to the fibre or in
   quiet periods may well have this much SNR.
2. **Too little signal: regularly missed footsteps pass QC.** At noise ×4,
   lane 1.5 m, recall is 54 % but every pass passes QC (stride RMSE 21 cm here,
   65 cm with another seed); at ×2, lane 3 m, a quarter pass with errors near
   280 cm. When every other footstep is missed, the step period is read as
   twice the true one; the remaining footsteps still lie on a clean line, so
   the v2.1 QC (≥ 8 steps, R² ≥ 0.95, ≥ 80 % inliers) cannot see it.

## Candidate fixes (checked, NOT applied — need approval as v2.2)
`candidate_fixes_check.py` post-processes v2.1 output (20 passes per setting):

| Setting | v2.1 | + dominance filter | + plausibility QC | both |
|---|---|---|---|---|
| Baseline | 0.49 cm (QC 100 %) | 0.49 | 0.49 | 0.49 |
| Noise ×0.5 | 16.8 cm (100 %) | **0.41 (100 %)** | 0.87 (95 %) | 0.41 (100 %) |
| Noise ×4, lane 1.5 m | 65.2 cm (100 %) | 65.2 | **2.8 (75 %)** | 2.8 (75 %) |
| Noise ×2, lane 3 m | 408 cm (45 %) | 408 | **8.1 (5 %)** | 8.1 (5 %) |

Stride RMSE (QC pass rate). Dominance filter: drop a detection more than 15 dB
below the envelope maximum within ±0.45 s. Plausibility QC: cadence 70–150
steps/min and step length 0.3–1.0 m. The plausibility ranges are a judgment
for walking; running intruders exceed 150 steps/min, so for security use the
gate should flag rather than discard. Both changes leave the baseline
unchanged. Decision needed before Phase A: adopt them as v2.2 (v2.1 stays
tagged as the simulation baseline).
