"""Test io_sim.py against a fake h5py that imitates what h5py returns for a
file written by MATLAB's h5create/h5write/h5writeatt: the array transposed
(column-major), numeric attributes as 1x1 float arrays, text as bytes."""
import sys, types
import numpy as np
import pytest


class FakeDataset:
    def __init__(self, arr, attrs):
        self._a, self.attrs = arr, attrs
    def __getitem__(self, key):
        return self._a


class FakeFile(dict):
    def __init__(self, store):
        super().__init__(store)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def install_fake_h5py(store):
    m = types.ModuleType("h5py")
    m.File = lambda path, mode="r": FakeFile(store[path])
    sys.modules["h5py"] = m


def matlab_style_file(nt=600, nch=161):
    X = np.arange(nt)[:, None] * 1000.0 + np.arange(nch)[None, :]   # X(t,c) = 1000 t + c
    attrs = {"fs": np.array([[500.0]]), "dx": np.array([[0.25]]),
             "gauge_length": np.array([[1.0]]), "x0": np.array([[0.0]]),
             "units": np.bytes_(b"strain"), "source": np.bytes_(b"sim"),
             "label_subject_id": np.bytes_(b"S01"), "label_pass_id": np.bytes_(b"S01_d1.0_r1"),
             "label_height_cm": np.array([[163.4]]), "label_lane_offset_m": np.array([[1.0]])}
    return X, {"data": FakeDataset(X.T.astype(np.float32), attrs)}   # h5py sees X'


def test_transpose_and_labels():
    X, f = matlab_style_file()
    install_fake_h5py({"a.h5": f})
    from das_height.io_sim import load_sim_h5
    rec = load_sim_h5("a.h5")
    assert rec.data.shape == X.shape
    assert rec.data[7, 3] == 7003                      # orientation correct
    assert rec.fs == 500.0 and rec.dx == 0.25 and rec.units == "strain"
    assert rec.labels["subject_id"] == "S01"
    assert rec.labels["height_cm"] == pytest.approx(163.4)


def test_wrong_orientation_is_caught():
    X, f = matlab_style_file(nt=600, nch=161)
    f["data"]._a = X.astype(np.float32)   # stored without the column-major flip
    install_fake_h5py({"b.h5": f})
    from das_height.io_sim import load_sim_h5
    with pytest.raises(ValueError):
        load_sim_h5("b.h5")


def test_missing_attribute_is_reported():
    X, f = matlab_style_file()
    del f["data"].attrs["fs"]
    install_fake_h5py({"c.h5": f})
    from das_height.io_sim import load_sim_h5
    with pytest.raises(ValueError, match="fs"):
        load_sim_h5("c.h5")
