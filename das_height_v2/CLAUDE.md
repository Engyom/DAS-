# Project rules for Claude (das_height_v2)

Claude Code reads this file at the start of every session in this folder.

## What this project is
DAS-based walker HEIGHT estimation, part of the PNU graduation project
"AI-Based DOFS Network for Autonomous Perimeter Security and Real-Time
Intruder Detection and Tracking: Hajj Season Use Case". Height only; weight
estimation is out of scope for now.
- matlab/make_height_dataset.m : MATLAB generator of SIMULATED passes (HDF5)
- das_height/                  : Python pipeline (footsteps, features, ridge/GPR, LOSO)
- tests/                       : pytest suite + tools; see CHANGES.md and README.md

## Hard rules
1. Work ONLY inside this folder (das_height_v2). Never read-modify-write the
   original das_height project or any file outside this folder.
2. Never edit matlab/make_height_dataset.m: it is a copy of the team's
   original MATLAB generator and must stay identical. MATLAB changes go in
   NEW files only. Never touch the team's phi-OTDR simulation.
3. Inspect before changing: read the relevant files and run the tests first.
4. Before any edit, state which files you will change, what and why, and
   wait for approval. One logical change at a time.
5. After every change run:  python -m pytest tests  and report the result.
   Never weaken or delete a test to make it pass; explain failures instead.
6. Label every result. Simulated results come from the assumed model
   step length = k x height and are NOT accuracy. Only results from real
   FEBUS recordings of measured volunteers may be called EXPERIMENTAL.
   train_eval.py must always be run with --label SIMULATED or EXPERIMENTAL.
7. Never invent accuracy figures, datasets, specifications or references.
8. Model inputs are FEATURE_NAMES in das_height/features.py. Never use
   true_step_length_m, height_cm or any label as an input.
9. Keep LOSO: every fold holds out all passes of one subject; fit scaling
   and hyperparameters inside the training fold only.

## Commands
- Install:     python -m pip install -r requirements.txt
- Tests:       python -m pytest tests
- Sim study:   python tests/synthetic_study.py 30 1.0 1.50 1.80 0.02
- Features:    python -m das_height.build_dataset --sim-dir sim_out --out features.csv
- Train/eval:  python -m das_height.train_eval --features features.csv --out results --label SIMULATED
- MATLAB->Python check: run matlab/export_roundtrip_test.m in MATLAB (scratch
  folder), then python tests/check_matlab_roundtrip.py roundtrip_test.h5

## Known untested parts
io_febus.py on any FEBUS file: test it before relying on it.
Verified (21 Nov 2026): io_sim.py on one real MATLAB-written file
(roundtrip test, 10/10 PASS, MATLAB R2026b); make_height_dataset.m runs
and io_sim.py reads all 180 of its files (0 failures). Nothing has been
validated on real DAS recordings.

## v2.1 conventions (see CHANGES.md)
- bandpass() uses mirror ("even") edge padding, 3 periods of f1 long.
  Do not go back to SciPy's default odd padding: it creates false
  footsteps at record edges.
- n_steps = validated footsteps (inliers of the robust fit);
  n_detections = all detections. Every gait feature uses the validated
  footsteps only.
