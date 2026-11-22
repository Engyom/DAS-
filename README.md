# DAS-based walker height estimation: simulation baseline v2.1

Graduation project, Princess Nourah bint Abdulrahman University (PNU),
Communications Engineering:
**"AI-Based Distributed Optical Fiber Sensing (DOFS) Network for Autonomous
Perimeter Security and Real-Time Intruder Detection and Tracking: Hajj
Season Use Case"**. Planned interrogator: FEBUS Optics A1 (not yet
available to the team).

> **STATUS: SIMULATION BASELINE, frozen 21 Nov 2026.** Every result in this
> repository comes from **simulated** DAS recordings. Nothing here has been
> validated on real DAS measurements, and the simulated accuracy is **not**
> a real-world performance figure. In the simulated data, height enters only
> through the assumed rule *step length = k x height*.

---

## 1. Overview and research objective
Part of the project's *human profiling* objective: estimate the **height** of
a person walking beside a buried or surface fibre from distributed acoustic
sensing (DAS, phi-OTDR) data. The pipeline:

DAS record -> band-pass filter -> footstep detection -> footstep
localisation along the fibre -> robust walking-line fit (validated
footsteps) -> gait features (step length, cadence, step-length variability)
-> regression (ridge, GPR) -> height, evaluated leave-one-subject-out.

The underlying hypothesis is that step length scales with height. This
repository provides a tested, documented pipeline and a **simulation
baseline** to compare against once real recordings exist. Weight estimation
is out of scope here.

Full technical description, assumptions, results and limitations:
**[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)**. Change history:
[das_height_v2/CHANGES.md](das_height_v2/CHANGES.md).

## 2. Repository structure
```
.
|-- README.md                  this file (handover guide)
|-- requirements-tested.txt    exact package versions the pipeline was run with
|-- docs/TECHNICAL_REPORT.md   pipeline, assumptions, fixes, results, limitations
|-- das_height_v2/             the project (source, tests, MATLAB generator)
|   |-- das_height/            Python package
|   |   |-- schema.py          DASRecord: common format for simulated and FEBUS data
|   |   |-- io_sim.py          read MATLAB-written HDF5 passes
|   |   |-- io_febus.py        read FEBUS A1 files via xdas (UNTESTED, no file yet)
|   |   |-- footsteps.py       band-pass, footstep detection and localisation
|   |   |-- features.py        robust line fit, validated footsteps, gait features
|   |   |-- build_dataset.py   folder of passes -> features.csv (+ pipeline check)
|   |   `-- train_eval.py      QC, LOSO evaluation of baseline/linear/ridge/GPR
|   |-- matlab/make_height_dataset.m   simulated-pass generator (do not edit; see CLAUDE.md)
|   |-- matlab/export_roundtrip_test.m MATLAB -> Python format check
|   |-- tests/                 pytest suite (22 tests) + simulated planning study
|   |-- requirements.txt       Python dependencies (unpinned)
|   |-- README.md, CHANGES.md  project notes and change log
|   `-- CLAUDE.md, .claude/settings.json  rules for Claude Code (plan mode;
|                              the generator is protected from edits)
`-- sim_scratch/               frozen SIMULATED results of v2.1
    |-- sim_out/manifest_truth.csv  truth labels of the 180 passes (the .h5 files
    |                              are NOT in git: ~1.7 GB, regenerate with MATLAB)
    |-- features.csv           feature table, 180 passes
    |-- features_pipeline_check.json  measured vs true step length
    |-- results/metrics.json   all metrics (label SIMULATED)
    |-- results/predictions.csv  LOSO predictions, one row per pass
    |-- results/height_scatter.png  estimated vs measured height
    |-- show_system.py         4-panel figure of one pass
    `-- system_view_S01_d1.0_r1.png  that figure for pass S01_d1.0_r1
```

## 3. Software and dependencies
- **Python 3.11 or newer** (run with 3.14.10 on Windows 11 and 3.13 on Linux)
- Packages (`das_height_v2/requirements.txt`): numpy, scipy, pandas,
  scikit-learn, matplotlib, h5py, pytest, and xdas (only for real FEBUS files).
  Exact tested versions: `requirements-tested.txt`.
- **MATLAB R2016b or newer, no toolboxes**: only needed to regenerate the
  simulated recordings (run with R2026b).
- Git.

## 4. Installation and environment setup
Windows (PowerShell):
```powershell
git clone <REPOSITORY-URL> das_height_project
cd das_height_project\das_height_v2
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Linux / macOS:
```bash
git clone <REPOSITORY-URL> das_height_project
cd das_height_project/das_height_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
To match the tested environment exactly, use
`python -m pip install -r ../requirements-tested.txt` instead (xdas can be
left out if you only work with simulated data).

**All commands below are run from `das_height_v2/` with the virtual
environment active**, unless stated otherwise.

## 5. Simulated dataset
The 180 simulated passes (30 subjects x 3 lanes x 2 repetitions, one HDF5
file each) are **not in the repository** because of their size; only their
truth manifest is. Regenerate them in MATLAB. The script writes `sim_out/`
into MATLAB's *current folder*, so start in `sim_scratch`:
```matlab
cd('<path-to>/das_height_project/sim_scratch')
run('../das_height_v2/matlab/make_height_dataset.m')   % -> "Wrote 180 passes to sim_out"
```
The generator is seeded (`rng(1)`). Check that the regenerated
`sim_out/manifest_truth.csv` matches the committed one, for example with
`git diff sim_scratch/sim_out/manifest_truth.csv`. Assumptions used (step
length = k x height, cadence independent of height, constant 70 kg,
ground and noise model) are listed in docs/TECHNICAL_REPORT.md, section 6.
Do not commit the `.h5` files (`.gitignore` excludes them).

Optional format check (MATLAB, in an empty scratch folder):
`run('<path-to>/das_height_v2/matlab/export_roundtrip_test.m')`, then
`python tests/check_matlab_roundtrip.py <scratch>/roundtrip_test.h5`, which should print 10/10 PASS.

## 6. Build the feature dataset
```bash
python -m das_height.build_dataset --sim-dir ../sim_scratch/sim_out --out ../sim_scratch/features.csv
```
Expected: `wrote 180 passes` and pipeline check
`step_length_MAE_cm` = 0.2543 (written to `../sim_scratch/features_pipeline_check.json`).

## 7. Run the height-estimation models
```bash
python -m das_height.train_eval --features ../sim_scratch/features.csv --out ../sim_scratch/results --label SIMULATED
```
This runs the mean baseline, step-length-only linear model, ridge and GPR
with leave-one-subject-out evaluation (about 1 minute) and writes
`predictions.csv`, `metrics.json` and `height_scatter.png`. Use
`--label EXPERIMENTAL` **only** for real recordings of measured volunteers.

## 8. Run the tests
```bash
python -m pytest tests
```
Expected: **31 passed** (22 pipeline + 9 integration). They need no MATLAB and no data files.
Simulated planning study (not a test):
`python tests/synthetic_study.py 30 1.0 1.50 1.80 0.02`.

## 9. Reproduce the reported metrics and figures
**A. Metrics from the committed features (no MATLAB needed).** Write to a
new folder so the frozen results are not overwritten:
```bash
python -m das_height.train_eval --features ../sim_scratch/features.csv --out ../sim_scratch/results_check --label SIMULATED
python -c "import json;a=json.load(open('../sim_scratch/results/metrics.json'));b=json.load(open('../sim_scratch/results_check/metrics.json'));print(a['subject_level']['ridge']==b['subject_level']['ridge'], round(b['subject_level']['ridge']['MAE_cm'],2))"
```
Expected: `True 5.85`. Ridge reproduces exactly. GPR may differ in the
5th-6th decimal between machines (optimiser round-off). Delete
`results_check` afterwards.

**B. Full chain:** section 5 (MATLAB) -> section 6 -> section 7, then
compare with the committed files using `git diff --stat sim_scratch`.

**C. System figure** (needs the regenerated `.h5` files). Run from the
**repository root**:
```bash
python sim_scratch/show_system.py --pass S01_d1.0_r1 --no-show
```
This writes `sim_scratch/system_view_S01_d1.0_r1.png`: panel 1 signal and
detections, panel 2 walking-line fit, panel 3 ridge LOSO results from
`results/predictions.csv`, panel 4 summary. Omit `--no-show` to open the
window. The script finds `das_height_v2/` next to `sim_scratch/` by itself
(`--project` overrides it).

**Frozen v2.1 results (SIMULATED, 30 subjects, 180 passes, subject level):**

| Model | MAE cm (95% CI) | RMSE cm | R^2 |
|---|---|---|---|
| Mean baseline | 7.71 (5.99-9.53) | 9.16 | -0.07 |
| Step length only | 5.88 (4.44-7.44) | 7.20 | 0.339 |
| **Ridge** | **5.85 (4.44-7.34)** | **7.09** | **0.358** |
| GPR | 6.80 (5.07-8.56) | 8.38 | 0.104 |

R^2 is computed from `predictions.csv`; the rest is in `metrics.json`.
These numbers show the pipeline recovering the simulation's own assumption;
they are **not** real-world accuracy.

## Integration with the team's MATLAB system (T12)
`das_height_v2/integration/` provides the agreed interface: `estimate_height(record)`
(11 fields: height_cm, uncertainty_cm, n_steps, qc_pass, qc_reason, speed_mps,
cadence_spm, stride_m, direction, model_version, data_label), models saved as JSON
(trained on the 180 SIMULATED passes), and stubs for `classify_event` and
`profile_person`. Details, limits and the MATLAB example: `das_height_v2/integration/README.md`.
Tests: `python -m pytest tests` (31 passed).

## 10. Current status, assumptions and limitations
- **Assumptions:** step length = k x height, with k ~ N(0.415, 0.02) per
  person; cadence 95-125 steps/min independent of height; constant weight;
  simple ground and noise model; one straight walker; FEBUS-like acquisition
  (500 Hz, 0.25 m channel spacing, 1 m gauge length).
- **Main limitations:** no real DAS data (the FEBUS reader is untested);
  height information is assumed, not measured; GPR intervals are too
  narrow (82% coverage); 84 interior false detections are rejected by the
  line fit but the detector was not changed; parameters (band, threshold,
  windows) are not calibrated to a real site; single walker only; weight is
  not studied.
  Full list: docs/TECHNICAL_REPORT.md, section 11.

## 11. Implemented, simulated, planned
| Implemented and tested (simulated data) | Simulated only | Planned |
|---|---|---|
| MATLAB generator, HDF5 I/O (verified) | All accuracy figures | Real FEBUS A1 recordings of measured volunteers (ethics approval, stadiometer) |
| Filtering, detection, localisation, robust inliers | Step length-height relation | Testing `io_febus.py` on a real file |
| Gait features, QC | Ground, noise and DAS response | Site calibration of band, threshold and windows |
| Ridge / GPR, LOSO, CIs, tests | | Multiple walkers, real-time streaming, weight (separate study) |

## 12. Recommended next steps
1. **Get one real FEBUS A1 file** and test `io_febus.py`: confirm the
   block size and overlap, units, scaling and distance axis with FEBUS Optics.
2. **Calibrate on site:** measure the footstep frequency band, noise floor
   and localisation error with a walker at known positions.
3. **Pilot experiment** following the simulated protocol (lanes 0.5, 1.0 and
   2.0 m, repeated passes; heights measured with a stadiometer; ethics
   approval). Run `build_dataset --febus-manifest ... --participants ...` and
   `train_eval --label EXPERIMENTAL`.
4. Compare the real step length-height relation with the assumed k, and
   update the generator's assumptions in a **new** MATLAB file, never by
   editing `make_height_dataset.m`.
5. Treat GPR uncertainties with caution (recalibrate) and add a
   multiple-comparison correction if several models are reported.
6. Keep every result labelled SIMULATED or EXPERIMENTAL, record changes in
   `das_height_v2/CHANGES.md`, and run `python -m pytest tests` after every change.

## Working rules (from das_height_v2/CLAUDE.md)
Do not edit `matlab/make_height_dataset.m` (MATLAB changes go in new files).
Never use labels (`height_cm`, `true_step_length_m`) as model inputs. Keep
LOSO with all fitting inside the training fold. Never present simulated
numbers as accuracy.

## Licence
No licence has been chosen yet, so default copyright applies: the code can
be viewed, but reuse needs the authors' permission. A licence will be added
once the project team agrees on one.
