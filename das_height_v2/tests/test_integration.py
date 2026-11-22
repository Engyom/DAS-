"""Tests for the T12 integration layer (integration/). SIMULATED data only."""
import json, math, os
import numpy as np
import pandas as pd
import pytest

from integration.estimate_height import (estimate_height, load_model, predict_height,
                                         OUTPUT_FIELDS, MODEL_DIR)
from integration import bridge
from integration.fit_sim_models import fit_ridge, fit_steplength_linear
from das_height.train_eval import qc_filter
from das_height.schema import DASRecord
from tests.sim_py import simulate_pass

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "sim_scratch", "features.csv")


@pytest.fixture(scope="module")
def walk():
    rng = np.random.default_rng(3)
    rec, L, cad = simulate_pass(1.70, 0.415, 1.0, 1.0, rng)
    return rec, L, cad


def test_output_has_exactly_the_agreed_fields(walk):
    rec, _, _ = walk
    out = estimate_height(rec.data, rec.fs, rec.dx, rec.gauge_length)
    assert tuple(out) == OUTPUT_FIELDS and len(out) == 11


def test_simulated_walk_passes_qc_and_recovers_gait(walk):
    rec, L, cad = walk
    out = estimate_height(rec.data, rec.fs, rec.dx, rec.gauge_length, data_label="SIMULATED")
    assert out["qc_pass"] and out["qc_reason"].startswith("ok")
    assert abs(out["stride_m"] - 2 * L) < 0.03
    assert abs(out["cadence_spm"] - cad) < 2.0
    assert abs(out["speed_mps"] - L * cad / 60) < 0.02
    assert out["direction"] == 1 and out["n_steps"] >= 8
    assert 140 < out["height_cm"] < 200 and out["uncertainty_cm"] > 0
    assert out["data_label"] == "SIMULATED" and out["model_version"].startswith("steplength_linear@")


def test_reversed_walk_gives_negative_direction(walk):
    rec, _, _ = walk
    out = estimate_height(rec.data[:, ::-1], rec.fs, rec.dx, rec.gauge_length)
    assert out["direction"] == -1 and out["qc_pass"]


def test_noise_only_fails_qc_with_nan_height():
    x = np.random.default_rng(0).standard_normal((6000, 161))
    out = estimate_height(x, 500.0, 0.25, 1.0)
    assert not out["qc_pass"] and math.isnan(out["height_cm"]) and math.isnan(out["uncertainty_cm"])
    assert out["qc_reason"]


def test_invalid_inputs_are_rejected(walk):
    rec, _, _ = walk
    with pytest.raises(ValueError):
        estimate_height(rec.data, rec.fs, rec.dx, rec.gauge_length, data_label="REAL")
    with pytest.raises(ValueError):
        estimate_height(rec.data[:, 0], rec.fs, rec.dx, rec.gauge_length)
    with pytest.raises(ValueError):
        estimate_height(rec.data, 0, rec.dx, rec.gauge_length)


@pytest.mark.skipif(not os.path.exists(FEATURES_CSV), reason="frozen sim_scratch/features.csv not present")
@pytest.mark.parametrize("name, fitter", [("steplength_linear", fit_steplength_linear), ("ridge", fit_ridge)])
def test_saved_models_reproduce_scikit_learn(name, fitter):
    df, _ = qc_filter(pd.read_csv(FEATURES_CSV))
    _, sk = fitter(df)
    saved = load_model(name)
    X = df[saved["features"]]
    mine = np.array([predict_height(saved, row) for row in X.to_dict("records")])
    assert np.allclose(mine, sk.predict(X.to_numpy(float)), atol=1e-6)


def test_saved_models_are_labelled_simulated():
    for name in ("steplength_linear", "ridge"):
        m = json.load(open(os.path.join(MODEL_DIR, f"{name}.json")))
        assert m["data_label"] == "SIMULATED" and m["pipeline_version"] == "v2.1-sim-baseline"


def test_bridge_matches_direct_call_and_stubs_are_marked(walk):
    rec, _, _ = walk
    record = {"data": rec.data, "fs": rec.fs, "dx": rec.dx, "gauge_length": rec.gauge_length,
              "x0": 0.0, "units": "strain", "data_label": "SIMULATED"}
    a = bridge.estimate_height(record)
    b = estimate_height(rec.data, rec.fs, rec.dx, rec.gauge_length)
    assert a == b or all((a[k] == b[k]) or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k]))
                         for k in OUTPUT_FIELDS)
    with pytest.raises(KeyError):
        bridge.estimate_height({"data": rec.data, "fs": rec.fs})
    assert bridge.classify_event(None) == {"class": "unknown", "confidence": 0.0}
    p = bridge.profile_person(None, 7)
    assert set(p) == {"person_id", "is_new", "confidence"} and p["confidence"] == 0.0
