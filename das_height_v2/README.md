# das_height v2: DAS-based walker height estimation (research prototype)

See CHANGES.md for what changed from v1, and for v2.1 (edge padding of the
band-pass filter; all gait features from the validated footsteps only).

STATUS: tested end to end on SIMULATED data only. No result on simulated
data is a measure of real-world accuracy. Height information in the
simulated data comes from one assumption (step length = k x height).

## Layout
    matlab/make_height_dataset.m   MATLAB: simulated passes -> sim_out/*.h5
    exploratory/load_identifiability/  T11 weight/load identifiability study (SIMULATED, no model)
    integration/                   estimate_height + saved SIMULATED models for the team's MATLAB system (see integration/README.md)
    das_height/schema.py           common DASRecord (sim and FEBUS)
    das_height/io_sim.py           read MATLAB HDF5 (transposes column-major)
    das_height/io_febus.py         read FEBUS A1 via xdas (UNTESTED)
    das_height/footsteps.py        band-pass, footstep detection, localisation
    das_height/features.py         step length, cadence, speed, variability
                                   (from validated footsteps; n_steps, n_detections)
    das_height/build_dataset.py    folder of recordings -> features.csv
    das_height/train_eval.py       ridge + GPR, LOSO, MAE/RMSE, plots
    matlab/export_roundtrip_test.m MATLAB: one known test file for the format check
    tests/sim_py.py                Python mirror of the MATLAB generator
    tests/test_*.py                pytest suite (run: python -m pytest tests)
    tests/check_matlab_roundtrip.py  verify Python reads a MATLAB-written file
    tests/synthetic_study.py       simulated planning study (not a test)

## Run
    pip install -r requirements.txt
    # 1. In MATLAB: run matlab/make_height_dataset.m  (writes sim_out/)
    python -m das_height.build_dataset --sim-dir sim_out --out features.csv
    python -m das_height.train_eval --features features.csv --out results --label SIMULATED
    # Real data later:
    python -m das_height.build_dataset --febus-manifest passes.csv \
           --participants participants.csv --out features_real.csv
    python -m das_height.train_eval --features features_real.csv --out results_real --label EXPERIMENTAL

    # Tests without MATLAB:  python -m pytest tests
    # Format check:  in MATLAB run export_roundtrip_test, then
    #                python tests/check_matlab_roundtrip.py roundtrip_test.h5
