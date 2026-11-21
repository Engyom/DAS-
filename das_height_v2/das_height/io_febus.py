"""Read real FEBUS A1 recordings with the open-source xdas library.

UNTESTED: written against the xdas documentation, without a FEBUS file.
The open call follows the documented example:
    xd.open("path.h5", engine="febus", vtype="tiles", overlaps=(100, 100), offset=1000)
whose overlaps/offset values are the docs' example for 2000-sample blocks.
Before use, confirm with FEBUS Optics the block size and overlap of your
firmware's files, the output units (strain or strain-rate) and scaling,
and the fibre distance of the test section.
https://xdas.readthedocs.io/en/stable/user-guide/io/data-formats.html
"""
import numpy as np
from .schema import DASRecord


def _coord_values(da, name):
    c = da[name]
    return np.asarray(getattr(c, "values", c))


def _seconds(t):
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return (t - t[0]).astype("timedelta64[ns]").astype(np.float64) * 1e-9
    return t.astype(np.float64) - float(t[0])


def _none_if_nan(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def load_febus(path, overlaps=(100, 100), offset=1000, x_start=None, x_stop=None,
               units="strain_rate", gauge_length=None, labels=None):
    import xdas as xd  # pip install xdas
    da = xd.open(path, engine="febus", vtype="tiles", overlaps=overlaps, offset=offset)
    x_start, x_stop = _none_if_nan(x_start), _none_if_nan(x_stop)
    if x_start is not None or x_stop is not None:
        da = da.sel(distance=slice(x_start, x_stop))
    da = da.load()
    if tuple(da.dims) == ("distance", "time"):
        da = da.transpose("time", "distance")
    if tuple(da.dims) != ("time", "distance"):
        raise ValueError(f"unexpected dimensions {da.dims}")
    t = _seconds(_coord_values(da, "time"))
    x = _coord_values(da, "distance").astype(np.float64)
    data = np.asarray(getattr(da, "values", da), dtype=np.float64)
    dt = np.median(np.diff(t))
    return DASRecord(data=data, fs=1.0 / dt, dx=float(np.median(np.diff(x))),
                     gauge_length=float(gauge_length) if gauge_length else np.nan,
                     x0=float(x[0]), units=units, labels=dict(labels or {}),
                     source="febus")
