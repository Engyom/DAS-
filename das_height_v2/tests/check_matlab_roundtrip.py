"""Check that Python reads a MATLAB-written HDF5 file correctly.

1. In MATLAB, in a scratch folder:  export_roundtrip_test
2. Then:  python tests/check_matlab_roundtrip.py path/to/roundtrip_test.h5
Needs h5py (pip install h5py). Prints PASS or the first failure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from das_height.io_sim import load_sim_h5


def main(path):
    rec = load_sim_h5(path, expect_channels=161)
    checks = [
        ("shape (time, channel) == (600, 161)", rec.data.shape == (600, 161)),
        ("X[7, 3] == 7003 (orientation)", rec.data[7, 3] == 7003),
        ("X[599, 160] == 599160", rec.data[599, 160] == 599160),
        ("fs == 500", rec.fs == 500.0),
        ("dx == 0.25", rec.dx == 0.25),
        ("gauge_length == 1", rec.gauge_length == 1.0),
        ("units == 'strain'", rec.units == "strain"),
        ("label subject_id == 'S01'", rec.labels.get("subject_id") == "S01"),
        ("label height_cm == 163.4", abs(rec.labels.get("height_cm", 0) - 163.4) < 1e-9),
        ("label true_step_length_m == 0.678",
         abs(rec.labels.get("true_step_length_m", 0) - 0.678) < 1e-9),
    ]
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    if not all(ok for _, ok in checks):
        sys.exit(1)
    print("All checks passed: MATLAB -> Python export works.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roundtrip_test.h5")
