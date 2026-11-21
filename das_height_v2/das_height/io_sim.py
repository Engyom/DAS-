"""Read HDF5 files written by make_height_dataset.m (MATLAB).

MATLAB stores arrays column-major, so a MATLAB [Nt x Nch] array appears
transposed ([Nch x Nt]) in h5py. We transpose it back here. Attributes
written by h5writeatt arrive as 1x1 arrays (numbers) or byte strings
(text); _attr() normalises both.

Tested here with a fake h5py that imitates MATLAB's layout
(tests/test_io_sim.py). Not yet tested on a real MATLAB-written file:
run matlab/export_roundtrip_test.m and tests/check_matlab_roundtrip.py.
"""
import numpy as np
from .schema import DASRecord

REQUIRED_ATTRS = ("fs", "dx", "gauge_length", "units")


def _attr(v):
    """Turn an h5py attribute value into a plain Python scalar or str."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return _attr(v.ravel()[0])
        if v.dtype.kind in "SU":                  # char array from MATLAB
            return "".join(_attr(c) for c in v.ravel())
        return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_sim_h5(path, expect_channels=None):
    import h5py  # pip install h5py
    with h5py.File(path, "r") as f:
        d = f["data"]
        a = {k: _attr(v) for k, v in d.attrs.items()}
        missing = [k for k in REQUIRED_ATTRS if k not in a]
        if missing:
            raise ValueError(f"{path}: missing attributes {missing}")
        data = np.asarray(d[()], dtype=np.float64).T      # -> (n_time, n_channel)
        labels = {k[6:]: v for k, v in a.items() if k.startswith("label_")}
        if "labels" in f:                                  # optional group form
            for k, v in f["labels"].items():
                labels[k] = _attr(v[()])
    rec = DASRecord(data=data, fs=float(a["fs"]), dx=float(a["dx"]),
                    gauge_length=float(a["gauge_length"]),
                    x0=float(a.get("x0", 0.0)), units=str(a["units"]),
                    labels=labels, source="sim")
    # Sanity check on orientation: a walking pass is much longer in time
    # samples than in channels; if not, the transpose is probably wrong.
    if expect_channels is not None and rec.data.shape[1] != expect_channels:
        raise ValueError(f"{path}: expected {expect_channels} channels, got {rec.data.shape[1]}")
    if rec.data.shape[0] < rec.data.shape[1]:
        raise ValueError(f"{path}: fewer time samples than channels "
                         f"{rec.data.shape}; check the MATLAB/HDF5 orientation")
    return rec
