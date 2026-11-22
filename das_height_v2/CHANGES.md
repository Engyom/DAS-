# Changes from das_height (v1) to das_height_v2

The original package (das_height.zip) and matlab/make_height_dataset.m are
unchanged (checksums verified). All edits are in this copy.

## Bugs and design problems fixed
1. features.py - Removed speed_mps from the model inputs. speed = step length
   x cadence / 60 exactly, so the three inputs were redundant; ridge then
   gave cadence and speed large, opposite, meaningless coefficients. Speed
   is still computed and saved as a descriptive column.
2. train_eval.py - Training rows are weighted 1 / (passes of that subject),
   so a subject with more passes no longer counts more (baseline,
   step-length model and ridge; GPR cannot take weights).
3. io_febus.py - Open call corrected to the documented form
   xd.open(path, engine="febus", vtype="tiles", overlaps=..., offset=...);
   data loaded with .load(); dimension order checked and transposed to
   (time, distance) if needed; datetime time axes converted to seconds;
   empty (NaN) crop values from the manifest no longer break .sel().
4. build_dataset.py - Relative file paths in passes.csv are resolved from
   the manifest's folder; duplicate subject IDs and missing heights are
   reported; one failed file no longer stops the whole run (error kept in
   an "error" column).
5. io_sim.py - Attribute parsing handles every form h5py returns for
   MATLAB attributes (1x1 arrays, numpy bytes, char arrays); missing
   attributes and a wrong (non-transposed) orientation raise clear errors.

## Improvements
6. footsteps.py - Threshold now = 8 x noise floor estimated across channels,
   instead of 4 x the 10th percentile in time. Both gave the same results in
   every simulated test (including cropped records and 95-190 steps/min);
   the new rule does not assume 10% of a recording is quiet. Returns
   diagnostics (noise floor, threshold, peaks found/kept); band validated.
7. train_eval.py - Added a step-length-only linear model as a second
   baseline; bootstrap 95% confidence intervals (over subjects) for MAE and
   RMSE; ridge alpha chosen in each fold is reported; ridge refitted on all
   data with coefficients in cm per unit (for explanation only, labelled
   as such); refuses to run with fewer than 4 subjects; checks required
   columns; QC thresholds are command-line options; --label SIMULATED or
   EXPERIMENTAL is now required and written into every output file.
8. build_dataset.py - For simulated data, measured vs true step length is
   checked and saved (features_pipeline_check.json). true_step_length_m is
   never a model input (tested).

## Tests added (pytest, 19 tests, all passing)
- tests/test_features.py   exact walk, missed steps, false detection,
                           reversed direction, too few steps
- tests/test_footsteps.py  step length at 0.5/1/2 m lanes, cropped record,
                           noise-only record, wrong orientation
- tests/test_io_sim.py     MATLAB layout via a fake h5py: transpose,
                           labels, wrong orientation, missing attribute
- tests/test_train_eval.py models beat baseline on a strong relation,
                           NO LABEL LEAKAGE (changing a subject's height does
                           not change that subject's own predictions),
                           weights, QC filter, minimum subjects
- tests/test_end_to_end.py simulated passes -> features -> command-line
                           training -> metrics, plot and labelled outputs

## New files for MATLAB -> Python verification (to run on your machine)
- matlab/export_roundtrip_test.m      writes one known test file
- tests/check_matlab_roundtrip.py     checks Python reads it exactly

## Renamed
- tests/run_synthetic_test.py -> tests/synthetic_study.py (pytest was
  collecting it as a test; it is a planning study, not a test)

## Not tested here (no MATLAB, h5py or xdas in this workspace)
- make_height_dataset.m and export_roundtrip_test.m in MATLAB
- io_sim.py on a real MATLAB-written file (tested with a fake h5py only;
  since verified, see below)
- io_febus.py on any FEBUS file

## Verified since
- io_sim.py on one real MATLAB-written file: matlab/export_roundtrip_test.m
  run in MATLAB R2026b, then python tests/check_matlab_roundtrip.py
  roundtrip_test.h5 -> 10/10 PASS (21 Nov 2026). This checks the file
  format, orientation and labels only; the output of make_height_dataset.m
  has not yet been checked.
- All 19 tests pass on Windows 11, Python 3.14.10, in a project .venv
  (21 Nov 2026).
- requirements.txt: added pytest.

# v2.1 (21 Nov 2026): two reliability fixes in the height pipeline

No change to the architecture, models, QC thresholds, inputs or
matlab/make_height_dataset.m. All numbers below are SIMULATED (the 180
passes / 30 subjects written by make_height_dataset.m); none is a
real-world accuracy.

## Fix 1 - footsteps.py, bandpass(): false detections at record edges
Root cause (measured): sosfiltfilt's default edge extension is "odd"
(point reflection about the first/last sample), 27 samples long. On a
noise-dominated edge the first sample is random, so odd reflection shifts
the extension by 2 x x[0]: a level step at the record boundary. The 2-60 Hz
band-pass step response rings for ~0.9 s inside the data, and the maximum
over 161 channels of these transients exceeded the threshold.
- Odd padding of 3 or 6 periods of f1 made it WORSE (180 passes with edge
  detections instead of 176): the step is at the boundary, so a longer
  extension cannot absorb it. The fix originally proposed (longer odd
  padding) was therefore rejected on evidence.
- Change: padtype="even" (mirror, no level step), padlen = 3 periods of
  f1 (750 samples at 2 Hz / 500 Hz), capped at record length - 1.
- Noise-only records, largest edge envelope / interior 99th percentile
  (1 = no artefact): odd-27 8.2, odd-750 8.8, even-27 1.20, even-750 1.06,
  even-1500 1.06. With a DC offset or linear drift: even-750 1.06-1.07;
  with a 0.5 Hz drift of 5 x noise: 1.59 (odd: 8-9).
- A fixed guard interval was not used: it would delete real footsteps.
  Real footsteps 0.1-0.4 s from an edge are all still detected (60/60 per
  margin, 30 simulated passes), with smaller position error than before
  (median 0.003 m vs 0.013 m at 0.2 s).
- Remaining limitation: a footstep still ringing when the record is cut
  (within ~0.4 s of the end) can give a second peak at the same position
  ~0.3 s later. The robust line fit rejects it and fix 2 keeps it out of
  every feature. When cutting walks from long FEBUS recordings, filter the
  long recording first or leave >= 1 s of margin. Footsteps within ~0.05 s
  of an edge may be missed (their energy peak lies outside the record).

## Fix 2 - features.py: all gait features from the same validated footsteps
Before: speed used the inliers of the robust position-time fit, but the
step period (cadence, step length), step_length_cv, step_length_direct_m
and n_steps used every detection. A false detection between two real steps
halved a step period and inflated step_length_cv.
- _robust_line returns the inlier mask; extract_pass_features keeps only
  the inliers before computing dt, T, cadence, step length, dl, cv.
- n_steps = validated footsteps (inliers); new QC column n_detections =
  all detections. fit_r2 and inlier_frac are unchanged in meaning. The
  inlier rule itself (3 x 1.4826 x MAD) is unchanged.
- QC in train_eval.py is unchanged in code; n_steps >= 8 now counts
  validated footsteps only (stricter). 0 of 180 passes rejected.

## Effect on the simulated results (before -> after both fixes)
| quantity (180 SIMULATED passes)          | before          | after           |
|------------------------------------------|-----------------|-----------------|
| detections within 0.2 s of record edges  | 302 (176 passes)| 0 (0 passes)    |
| detections / validated footsteps         | 9360 / -        | 9059 / 8975     |
| step length vs truth: MAE / max (cm)     | 0.374 / 2.81    | 0.254 / 0.92    |
| step_length_cv: median / p95 / max       | 0.074/2.59/3.45 | 0.069/0.185/0.24|
| passes with step_length_cv > 0.2         | 24              | 2               |

| subject level, LOSO (SIMULATED) | MAE cm (95% CI)    | RMSE cm     | R2            | p vs baseline |
|---------------------------------|--------------------|-------------|---------------|---------------|
| baseline_mean                   | 7.71 (unchanged)   | 9.16        | -0.07         | -             |
| steplength_linear               | 5.86 -> 5.88       | 7.18 -> 7.20| 0.342 -> 0.339| 0.037 -> 0.040|
| ridge                           | 5.85 -> 5.85 (4.44-7.34) | 7.10 -> 7.09 | 0.357 -> 0.358 | 0.029 -> 0.035 |
| gpr                             | 6.63 -> 6.80 (5.07-8.56) | 8.24 -> 8.38 | 0.134 -> 0.104 | 0.208 -> 0.185 |
GPR 95% interval coverage 0.83 -> 0.82 (still overconfident). R2 = 1 -
SSE/SST on subject-mean predictions (computed from predictions.csv; not
written by train_eval.py). Paired two-sided Wilcoxon, before vs after, per
subject |error|: ridge p = 0.84, GPR p = 0.17, step-length model p = 0.34:
no significant change. Height accuracy does not improve because in this
simulation height enters only through step length = k x height and the
per-subject spread of k sets the error; cadence and variability carry no
height information by design. The fixes make the features correct, not
the simulated accuracy better. Label-permutation check (heights shuffled
between subjects): all models no better than the mean baseline (no leakage).
The earlier v2 results (ridge 5.85 cm, 95% CI 4.41-7.34) are superseded.

## Tests (22, all passing)
Added: test_no_false_detections_at_record_edges, test_footstep_near_record_
start_is_kept (test_footsteps.py), test_false_detection_between_steps_does_
not_change_gait_features (test_features.py). The first and third fail on
the v2 code (verified); the second is a guard and passes on both.

## Other verifications
- make_height_dataset.m ran in MATLAB R2026b (180 passes); io_sim.py read
  all 180 files on Windows with h5py, 0 failures. (This verifies that the
  generator runs and its output is read correctly, not its physics.)
- The v2 baseline was reproduced exactly from those 180 files in a second
  environment (features identical to 1e-14) before any change was made.

## Not changed (found, outside the scope of these fixes)
- In the final features, 84 detections in 52 of 180 passes are rejected
  by the robust fit (n_detections - n_steps): 24 are extra peaks halfway
  between two real steps (mostly 0.5 m lane, slow cadence) and 60 are real
  footsteps localised slightly off the walking line. Fix 2 keeps all of
  them out of the features; the detector itself was not changed.

# integration v0.1 (22 Nov 2026): T12, interface for the team's MATLAB system

New files only; `das_height/`, `matlab/` and the frozen v2.1 results are unchanged.

- `integration/fit_sim_models.py`: fits `steplength_linear` and `ridge` on all
  180 SIMULATED passes (`../sim_scratch/features.csv`, same QC and subject
  weights as `train_eval.py`) and saves them as JSON in `integration/models/`.
  The saved ridge equals the v2.1 full fit in `metrics.json` (alpha 0.001,
  intercept 95.22 cm, 88.61 cm per m of step length).
- `integration/estimate_height.py` and `integration/bridge.py`: the agreed
  interface. `estimate_height` returns 11 fields (height_cm, uncertainty_cm,
  n_steps, qc_pass, qc_reason, speed_mps, cadence_spm, stride_m, direction,
  model_version, data_label). Default model: step length only (ridge also uses
  step-length CV, which is noise at coarse spacing or on real data).
  `uncertainty_cm` = LOSO pass-level RMSE on SIMULATED data (7.22 / 7.16 cm).
  `classify_event` and `profile_person` are clearly marked STUBS.
- `tests/test_integration.py`: 9 tests (fields, QC pass/fail, gait recovery,
  direction, input checks, JSON models reproduce scikit-learn, labels, bridge).
  Full suite: 31 passed (Linux, Python 3.13).
- `integration/matlab_example.m`: MATLAB call example, **untested in MATLAB**.
- Limits: needs a walk parallel to the fibre with >= 8 validated footsteps
  (fibre crossings fail QC); oblique paths shorten stride by cos(angle); models
  are SIMULATED only.

# exploratory T11 (22 Nov 2026): load-change / weight-range identifiability

New folder `exploratory/load_identifiability/`; no pipeline, model or result
of v2.1 or integration v0.1 is changed. SIMULATED + ASSUMED, no model trained.
- Error-propagation study of footstep amplitude at 0.4 m channel spacing
  (gauge length 0.8 / 1.2 m): lane distance, repeatability and crossing
  geometry measured from the generator equations; coupling, speed and
  footwear swept as labelled assumptions.
- Result: between people, weight ranges are not identifiable without
  calibration (1 class across 45–110 kg); with lane ±0.1 m and coupling
  calibrated to 10 %, about 3 broad classes; within one person a ~12 %
  amplitude change (≈ +8–10 kg) is detectable in the best case. Crossing
  intruders give no separable weight information.
- `tests/test_load_identifiability.py`: 6 tests. Full suite: 37 passed.

# T9 (22 Nov 2026): Phase A gait analysis tooling

New folder `gait_eval/`; the frozen pipeline, integration layer and earlier
results are unchanged. `tests/sim_py.py` is not modified (ground truth is
recovered by replaying its random draws).
- Footstep matching (dry-run offset, one-to-one +/-50 ms, tolerance sweep),
  repeated-measures Bland-Altman with MOVER CIs, proportional bias,
  repeatability coefficient.
- SIMULATED tooling check (4 participants x 3 lanes x 3 passes, 0.4 m / 0.8 m):
  recall 99.9 % / precision 99.1 % at 50 ms (84 % at 25 ms); stride LoA within
  the provisional thresholds; cadence LoA meets +/-2 spm only as a point
  estimate (far CI bound 2.6 spm). Findings: per-lane offset needed; D3 must
  say point estimate vs CI bound; +0.2 m localisation shift is a generator
  artefact of even gl/dx averaging (Python mirror), not a pipeline error.
- `tests/test_gait_eval.py`: 6 tests. Full suite: 43 passed.

# T10 (22 Nov 2026): gait sensitivity study (SIMULATED)

`gait_eval/sensitivity.py`, `gait_eval/candidate_fixes_check.py`, results in
`gait_eval/results_sensitivity/`; pipeline unchanged.
- Stride error SD <= 1.7 cm for spacing 0.25-2 m and lanes 0.5-2 m; works to
  lane 3 m at default noise; cadence 90-140 fine; noise cliff at detection
  SNR ~9-12 dB (threshold 8 x floor = 9 dB).
- Two v2.1 failure modes, both passing QC: (1) high SNR (>~33 dB peak):
  band-pass ringing side-lobes (-24 dB, -0.28/+0.41 s) detected as footsteps,
  cadence doubled; (2) low SNR: regularly missed footsteps read as a doubled
  step period.
- Candidate fixes checked, not applied: dominance filter (-15 dB within
  +/-0.45 s) and a cadence/step-length plausibility QC. Need approval as v2.2.
- 3 new tests; full suite 46 passed.

# v2.2 (22 Nov 2026): fixes for the two T10 failure modes

Approved by the project author. v2.1 stays in Git (tag v2.1-sim-baseline).
- footsteps.detect_footsteps: dominance filter (dominance_db=15,
  dominance_win_s=0.45): a peak more than 15 dB below the envelope maximum
  within +/-0.45 s is dropped (band-pass ringing side-lobes). New diagnostic
  n_dominated. dominance_db=None reproduces v2.1.
- features.gait_plausible + PLAUSIBLE_CADENCE_SPM (70-150) and
  PLAUSIBLE_STEP_M (0.3-1.0); train_eval.qc_filter(plausibility=True) and
  integration.estimate_height reject implausible passes with a reason, but
  estimate_height still returns their gait fields (running is flagged).
- features: new QC value fill_ratio = validated footsteps / (1 + span /
  step length); QC requires fill_ratio >= 0.85 (MIN_FILL_RATIO; judgment
  value, to re-set on Phase A data). Added because the plausibility gate
  alone catches only regularly missed steps that push cadence below
  70 steps/min.
- T10 rerun with v2.2 (gait_eval/results_sensitivity_v2.2/): the high-SNR
  failure is gone (noise x0.5: stride RMSE 38.5 -> 0.4 cm; precision at lane
  0.5 m 94-99.5 % -> 99-100 %); spacing 1-2 m at lane 0.5 m improves
  (1.7 -> 0.6 and 2.5 -> 1.1 cm). The low-SNR failure is reduced, not
  removed: in the grid cell noise x4 / lane 1.5 m QC now rejects 11 of 12
  passes, but the one it keeps has a 67 cm error. Every cell where this
  happens has a median detection SNR below 10 dB; a per-pass SNR gate is
  left for after Phase 0, when the real SNR is known.
- Effect on the 180 simulated passes: 8 passes change (all at lane 0.5 m,
  4-12 side-lobe detections removed each); pipeline step-length MAE
  0.254 -> 0.247 cm; QC rejects 0. LOSO results essentially unchanged
  (ridge subject-level MAE 5.85 cm, RMSE 7.09 -> 7.10; step-length-only
  5.88 -> 5.89 / 7.20 -> 7.21; GPR 6.80 -> 6.69). New files:
  sim_scratch/features_v2.2.csv, sim_scratch/results_v2.2/ (v2.1 files kept).
- integration/models refitted on the v2.2 features (model_version
  ...@v2.2-sim; step-length model 104.53 + 89.12 cm/m, was 104.44 + 89.28).
  S01_d1.0_r1.h5 still gives 168.0 cm (+/- 7.2).
- Tests: +4 (dominance filter, plausibility QC, fill-ratio QC, estimate_height flag);
  test_integration now checks the v2.2 models. Full suite: 50 passed.
