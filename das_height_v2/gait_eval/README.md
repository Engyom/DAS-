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
