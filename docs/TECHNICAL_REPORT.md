# Technical report: DAS walker-height estimation, simulation baseline v2.1

Frozen on 21 Nov 2026. Part of the PNU graduation project "AI-Based
Distributed Optical Fiber Sensing (DOFS) Network for Autonomous Perimeter
Security and Real-Time Intruder Detection and Tracking: Hajj Season Use Case".

> **Every result in this document is SIMULATED.** The system has not been run
> on real DAS (FEBUS A1) recordings. In the simulated data, height enters
> only through an assumed rule (step length = k x height). The accuracy
> figures show that the pipeline recovers that assumption; they are **not**
> estimates of real-world performance.

Source of truth for every number below: `sim_scratch/results/metrics.json`,
`sim_scratch/results/predictions.csv`, `sim_scratch/features.csv`,
`sim_scratch/features_pipeline_check.json`. Change history:
`das_height_v2/CHANGES.md`.

---

## 1. Pipeline overview

```
 DAS record (time x channel, strain or strain-rate; fs, dx, gauge length)
   |  io_sim.load_sim_h5 (MATLAB HDF5)  |  io_febus.load_febus (FEBUS A1, UNTESTED)
   v
 DASRecord (schema.py)
   |  footsteps.bandpass        zero-phase Butterworth 2-60 Hz, mirror edge padding
   |  footsteps.detect_footsteps energy envelope -> threshold -> peaks
   |                            -> energy-centroid position of each footstep
   v
 footstep times [s] and positions [m] (all detections)
   |  features._robust_line     line fit position = v*t + c, MAD outlier rejection
   |  features.extract_pass_features  (validated inliers only)
   v
 one feature row per pass: step_length_m, cadence_spm, step_length_cv
   + descriptive speed_mps + QC: n_steps, n_detections, fit_r2, inlier_frac
   |  build_dataset             all passes -> features.csv (+ pipeline check, sim only)
   v
 train_eval: QC filter -> leave-one-subject-out (LOSO) evaluation of
   mean baseline, step-length-only linear model, ridge, GPR
   -> predictions.csv, metrics.json, height_scatter.png (all labelled)
```

## 2. Input data and schema
`DASRecord` holds `data` (n_time x n_channel), `fs` [Hz], `dx` [m],
`gauge_length` [m], `x0` [m], `units`, `labels` (subject_id, pass_id,
height_cm, ...) and `source`.
- **Simulated:** `io_sim.load_sim_h5` reads one HDF5 file per pass written
  by MATLAB (`/data` dataset + attributes). MATLAB is column-major, so the
  array is transposed back to (time, channel). The orientation and the
  required attributes are checked. Verified on a MATLAB-written test file
  (`export_roundtrip_test.m`, 10/10) and on all 180 generated files.
- **Real (planned):** `io_febus.load_febus` uses `xdas`
  (`xd.open(path, engine="febus", vtype="tiles", overlaps=..., offset=...)`).
  **Untested: no FEBUS file has been available.** Block size, overlap, units
  and scaling must be confirmed with FEBUS Optics before use.

## 3. Signal filtering and footstep detection (`footsteps.py`)
1. **Band-pass:** 4th-order Butterworth, 2-60 Hz (upper edge capped at
   0.45 fs), applied forward and backward (`sosfiltfilt`, zero phase).
   **Edge handling (v2.1):** mirror (`even`) padding, 3 periods of the low
   edge long (750 samples at 2 Hz / 500 Hz), capped at the record length - 1.
2. **Energy map:** squared filtered signal, moving average over 0.1 s per channel.
3. **Envelope:** maximum of the energy map over channels at each instant.
4. **Threshold:** 8 x noise floor, where the noise floor is the median over
   time of the median over channels (most channels are far from the walker
   at any instant). It does not assume a quiet part of the record.
5. **Peaks:** `find_peaks` above the threshold, at least 0.3 s apart.

Units are left native (strain or strain-rate). Differentiating white noise
would amplify high frequencies and can hide the footsteps.

## 4. Footstep localisation and robust inlier selection
- **Localisation:** for each peak, energy per channel over +/- 0.08 s. Take
  the channel of maximum energy, a window of +/- 4 m around it, subtract the
  median channel energy (noise floor), and use the energy-weighted centroid
  as the footstep position. A centroid is used rather than the maximum
  because DAS senses axial strain only: a footstep beside the fibre gives a
  weak "broadside" response at the nearest channel and a two-lobed pattern
  along the fibre, roughly symmetric about the footstep.
- Detections within 2 m of either end of the section are discarded.
- **Robust inlier selection (`features._robust_line`):** least-squares line
  position = v t + c. Residuals larger than 3 x 1.4826 x MAD are outliers,
  and the line is refitted on the inliers. R^2 is computed on the inliers.
  The inliers are the **validated footsteps**; every gait feature uses them
  only (v2.1).

## 5. Gait features (`features.py`)
From the n_steps validated footsteps (needs >= 6 detections and >= 6 inliers):
- `speed_mps` = |v| (descriptive only, not a model input)
- step period T = median of consecutive time differences, ignoring gaps
  > 1.5 x the median (missed or rejected steps)
- `cadence_spm` = 60 / T
- `step_length_m` = speed x T (averages over all validated steps; robust)
- `step_length_cv` = std / mean of |dx| between consecutive validated
  footsteps (dt < 1.5 T)
- QC: `n_steps` (validated), `n_detections` (all), `fit_r2`,
  `inlier_frac` (= n_steps / n_detections), `step_length_direct_m` (median |dx|)

**Model inputs: `step_length_m`, `cadence_spm`, `step_length_cv`.** Speed is
excluded: it equals step length x cadence / 60 exactly, and including all
three made the ridge coefficients unstable.

## 6. Simulated dataset (`das_height_v2/matlab/make_height_dataset.m`)
Seeded (`rng(1)`). 30 subjects x 3 lanes (0.5, 1.0, 2.0 m from the fibre) x
2 repetitions = **180 passes**. Output: `sim_out/*.h5` and
`sim_out/manifest_truth.csv` (truth labels; the manifest is in this repository).

**Assumptions (to be replaced with measured or cited values):**

| Item | Value used |
|---|---|
| Height | uniform 1.50-1.80 m (generated: 150.1-179.9 cm, mean 166.1) |
| **Step length vs height** | **L = k x H**; k per subject ~ N(0.415, 0.02), fixed across that subject's passes; x (1 + N(0, 0.01)) per pass |
| Cadence | uniform 95-125 steps/min per pass, **independent of height** |
| Weight | constant 70 kg (weight is not studied; no weight information in the data) |
| Footstep force | 1.15 x weight x g x (1 + N(0, 0.05)) |
| Footstep timing / sides | 0.01 s jitter; alternating left/right +/- 0.08 m about the lane |
| Path | straight, parallel to the fibre, from x = 3 m to 37 m of a 40 m section |
| Ground response | half-sine contact (0.12 s) convolved with a 25 Hz, 15%-damped oscillator; wave speed 200 m/s; attenuation exp(-0.15 r); burial depth 0.3 m |
| DAS response | axial strain of a radial wave: du_r/dr cos^2(theta) + (u_r/r) sin^2(theta); gauge-length average |
| Acquisition | fs 500 Hz, dx 0.25 m, gauge length 1 m, 161 channels |
| Noise | white Gaussian, 5% of a reference footstep per dx segment |

Because height enters only through L = k H, the best possible error is set
by the between-person spread of k (sd 0.02 / 0.415, about 4.8% of height).
Cadence and step-length variability carry **no** height information by
construction.

**Not modelled:** real soil and installation coupling, fading, polarisation
and other phi-OTDR noise, footwear, carried loads, speed changes, curved
paths, several walkers, environmental noise (traffic, crowds), and real
interrogator scaling.

## 7. Feature dataset (`build_dataset.py`)
`python -m das_height.build_dataset --sim-dir sim_out --out features.csv`
reads every pass, runs detection and feature extraction, attaches the
labels, and continues past failed files (error column). For simulated data
it compares measured with true step length (`true_step_length_m`, never a
model input) and writes `features_pipeline_check.json`.
Final check: 180 passes, step-length MAE **0.254 cm**, bias -0.047 cm, max 0.92 cm.

## 8. Training and evaluation (`train_eval.py`)
- **QC filter:** n_steps >= 8, fit_r2 >= 0.95, inlier_frac >= 0.8, no
  missing features. 0 of 180 passes rejected.
- **Models:** mean baseline; step-length-only linear regression; ridge
  (standardised; alpha from 13 values 1e-3...1e3 by inner LOSO); GPR
  (standardised; constant x ARD-RBF + white-noise kernel, normalised target).
- **Evaluation:** outer leave-one-subject-out (all 6 passes of a subject
  held out). Scaling, alpha and kernel are fitted inside the training fold
  only. Training rows are weighted 1 / (passes of that subject); GPR cannot
  take weights.
- **Reported:** pass-level and subject-level (mean of a subject's pass
  predictions) MAE, RMSE, bias and max error; bootstrap 95% CIs over
  subjects; a one-sided Wilcoxon test of each model's subject |error|
  against the baseline; GPR 95% interval coverage; ridge refitted on all
  data, for interpretation only.
- `--label SIMULATED|EXPERIMENTAL` is required and written into every output.
- R^2 below is computed from `predictions.csv` (1 - SSE/SST); `train_eval`
  does not write it.

## 9. Final simulation results (v2.1, SIMULATED, 30 subjects / 180 passes)

| Model | Subject MAE cm (95% CI) | Subject RMSE cm (95% CI) | Subject R^2 | p vs baseline | Pass MAE / RMSE cm | Pass R^2 |
|---|---|---|---|---|---|---|
| Mean baseline | 7.71 (5.99-9.53) | 9.16 (7.39-10.78) | -0.07 | - | 7.71 / 9.16 | -0.07 |
| Step length only | 5.88 (4.44-7.44) | 7.20 (5.52-8.81) | 0.339 | 0.040 | 5.90 / 7.22 | 0.334 |
| **Ridge** | **5.85 (4.44-7.34)** | **7.09 (5.51-8.55)** | **0.358** | **0.035** | 5.87 / 7.16 | 0.346 |
| GPR | 6.80 (5.07-8.56) | 8.38 (6.47-10.11) | 0.104 | 0.185 | 6.82 / 8.44 | 0.090 |

- Ridge: bias +0.01 cm, max subject error 15.0 cm. Fitted on all data
  (interpretation only): alpha 0.001, +88.6 cm per metre of step length.
- GPR 95% interval coverage 0.82: its intervals are too narrow.
- Ridge and the step-length-only model are practically equal. The gain over
  the baseline comes from step length, as the simulation assumes.
- **Leakage check:** with heights shuffled between subjects, every model is
  no better than the baseline (ridge MAE 7.72-7.79 cm, p >= 0.59).

## 10. The two v2.1 fixes and their impact
**Fix 1: false detections at record edges (`footsteps.bandpass`).**
SciPy's default edge extension is short (27 samples) and *odd* (point
reflection about the first and last sample). On a noise-dominated edge, that
shifts the extension by 2 x x[0], so there is a step at the boundary. The
band-pass rings for about 0.9 s inside the data, and the maximum over 161
channels crossed the threshold. Longer odd padding made it worse (the step
is at the boundary itself), so the fix proposed before testing was
rejected. Mirror padding of 3 periods of f1 removes the artefact: on
noise-only records, the largest edge envelope falls from 8-9x to 1.06x of
the largest values inside the record. It stays at 1.06-1.07x with a DC
offset or linear drift, and 1.59x with a strong 0.5 Hz drift. Real footsteps
0.1-0.4 s from an edge are still detected (60/60).

**Fix 2: consistent gait features (`features.extract_pass_features`).**
Before, speed used the inliers, but the step period, cadence, step length,
variability and step count used every detection. A false detection between
two steps halved one step period and inflated the variability. Now all of
them use the validated inliers, `n_steps` counts the inliers, and
`n_detections` keeps the original count.

| 180 SIMULATED passes | v2 | v2.1 |
|---|---|---|
| Detections within 0.2 s of record edges | 302 (176 passes) | **0** |
| Detections / validated footsteps | 9360 / - | 9059 / 8975 |
| Step length vs truth: MAE / max | 0.374 / 2.81 cm | 0.254 / 0.92 cm |
| step_length_cv: median / max | 0.074 / 3.45 | 0.069 / 0.24 |
| Passes with step_length_cv > 0.2 | 24 | 2 |
| Ridge subject MAE / RMSE / R^2 | 5.85 / 7.10 / 0.357 | 5.85 / 7.09 / 0.358 |

Height accuracy is unchanged (paired Wilcoxon, ridge p = 0.84, GPR p = 0.17).
In this simulation the error is set by the spread of k, so the fixes make
the features correct without changing the simulated accuracy.
Tests: 22/22 pass, including 3 regression tests for the fixes.

## 11. Remaining limitations and unresolved issues
1. **No real DAS data.** All results are simulated, and `io_febus.py` is untested.
2. **Height information is assumed**, not measured: L = k x H with an assumed
   k distribution. Real gait (speed-dependent step length, footwear, loads,
   crowds) is not represented.
3. **GPR is overconfident** (82% coverage for nominal 95%). Prefer ridge, or
   recalibrate GPR, before reporting its uncertainties.
4. **Detector false positives inside the record:** 84 detections in 52 passes
   are rejected by the line fit: 24 extra peaks halfway between two steps
   (mostly 0.5 m lane, slow cadence) and 60 real footsteps localised slightly
   off the line. They are kept out of the features, but the detector itself
   was not changed.
5. **Records cut close to a footstep:** a footstep still ringing at the
   end of a record can give a second peak at the same position about 0.3 s
   later; the line fit rejects it. When cutting walks from long FEBUS
   recordings, filter the long recording first or leave >= 1 s of margin.
6. **Fixed parameters:** the 2-60 Hz band, the 8x threshold, the 0.3 s
   separation and the 4 m window were chosen for the simulation. Measure them
   at the real test site.
7. **Single straight walker only:** no tracking of several walkers,
   crossing paths or direction changes.
8. **No multiple-comparison correction** across the three model-vs-baseline
   tests.
9. **Weight is out of scope:** the simulation uses a constant 70 kg, and no
   weight experiment is included here.
10. **The `show_system.py` default project path** is the original author's
    Windows path; pass `--project das_height_v2` (see README).

## 12. Status: implemented, simulated, planned
| Item | Status |
|---|---|
| MATLAB simulated-pass generator (HDF5 export) | Implemented, run (MATLAB R2026b) |
| MATLAB -> Python HDF5 reading | Implemented, verified (roundtrip 10/10, 180/180 files) |
| Filtering, detection, localisation, robust inliers, features | Implemented, tested on simulated data |
| Ridge / GPR with LOSO, baselines, CIs, tests | Implemented, tested on simulated data |
| Height accuracy | **Simulated only** |
| FEBUS A1 reader (xdas) | Implemented, **untested** (no file yet) |
| Real-site parameter calibration | Planned |
| Real data collection with measured volunteers (ethics approval, stadiometer) | Planned |
| Weight estimation | Planned / out of scope of this baseline |
| Real-time streaming (FEBUS ZeroMQ) | Planned |

## 13. Workspace note on how v2.1 was verified
The v2.1 features and results were produced from Yomna's 180 MATLAB-written
HDF5 files, in a Linux environment without `h5py`. Those files were read
through `io_sim.py` using a minimal read-only HDF5 stand-in. It was checked
by reproducing the v2 features generated on Windows with real `h5py`
exactly (max difference 1.8e-14) before any change was made. The stand-in is
not part of this repository. A teammate with `h5py` installed reads the
files directly.
