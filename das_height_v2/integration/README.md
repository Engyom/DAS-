# integration/ — interface for the team's MATLAB system (T12)

New layer on top of the das_height pipeline (v2.2 since 22 Nov 2026). **It does not change `das_height/`**;
it imports it. Everything here is trained and tested on **SIMULATED** data only.

| File | Purpose |
|---|---|
| `bridge.py` | The three agreed functions: `classify_event` (STUB), `profile_person` (STUB), `estimate_height(record)` |
| `estimate_height.py` | Runs the v2.2 pipeline on one pass and returns the 11 agreed fields |
| `fit_sim_models.py` | Fits the models on all 180 simulated passes and saves them as JSON |
| `models/steplength_linear.json` | Default model: height = a + b x step length |
| `models/ridge.json` | Optional model on all three features (same as the ridge full fit of the v2.2 run) |
| `matlab_example.m` | How to call it from MATLAB (tested on R2026b + Python 3.14, 22 Nov 2026) |

## estimate_height output (11 fields)

| Field | Meaning |
|---|---|
| `height_cm` | Estimated height; **NaN when QC fails** |
| `uncertainty_cm` | Typical error from LOSO on SIMULATED data (7.22 cm default model, 7.16 cm ridge); not per person |
| `n_steps` | Validated footsteps |
| `qc_pass`, `qc_reason` | QC: >= 8 validated footsteps, line fit R2 >= 0.95, inlier fraction >= 0.8. `qc_reason` says why it failed, or `ok` (with a note if a feature is outside the simulated training range) |
| `speed_mps`, `cadence_spm` | Walking speed along the fibre, steps per minute |
| `stride_m` | Stride length = speed x stride period (2 x step period) |
| `direction` | +1 towards increasing fibre distance, -1 decreasing, 0 unknown |
| `model_version` | e.g. `steplength_linear@v2.2-sim` |
| `data_label` | `SIMULATED` or `EXPERIMENTAL`, passed through from the input |

**Input record:** `data` (time x channel), `fs` [Hz], `dx` [m], `gauge_length` [m],
`x0` [m], `units` (`strain` / `strain_rate`), `data_label`.

## Limits

- Needs a walk roughly **parallel** to the fibre with >= 8 validated footsteps.
  A person **crossing** the fibre fails QC and gets no height: use the camera.
- A path at angle phi to the fibre shortens the estimated stride by cos(phi).
- Default model uses step length only: the ridge model also uses step-length
  variability (CV), which becomes noise at coarse channel spacing or on real data.
- On real (EXPERIMENTAL) data, the output is a **sim-to-real test** of a model
  trained on simulation, not a validated measurement.

## Use

```bash
# from das_height_v2/ (Python 3.13, requirements.txt installed)
python -m pytest tests/test_integration.py
python -m integration.fit_sim_models --features ../sim_scratch/features_v2.2.csv \
       --metrics ../sim_scratch/results_v2.2/metrics.json --out integration/models   # regenerate models
```

From full-das (git submodule of this repository, pinned to a commit): put the
submodule's `das_height_v2/` folder on `sys.path`, then
`from integration.bridge import classify_event, profile_person, estimate_height`.
In MATLAB, see `matlab_example.m` (`pyenv(Version="3.13", ExecutionMode="OutOfProcess")`).
