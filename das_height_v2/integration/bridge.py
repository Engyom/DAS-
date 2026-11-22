"""The three functions the team's MATLAB system calls (agreed interface).

Copy or import this module from full-das/ai/. MATLAB passes a scalar
struct, which arrives here as a Python dict.

    classify_event(signal_window) -> {class, confidence}            STUB
    profile_person(image, event_id) -> {person_id, is_new, confidence}  STUB
    estimate_height(record) -> 11 fields (see integration.estimate_height)

The two stubs return fixed, clearly marked values so the MATLAB pipeline
runs end to end until their owners replace them. Do not report their
outputs as results.
"""
from .estimate_height import estimate_height as _estimate_height, OUTPUT_FIELDS

_RECORD_KEYS = ("data", "fs", "dx", "gauge_length", "x0", "units", "data_label")


def classify_event(signal_window):
    """STUB: event classification is owned by another team member."""
    return {"class": "unknown", "confidence": 0.0}


def profile_person(image, event_id):
    """STUB: re-identification (DAS track + appearance + time/space limits) is not built yet."""
    return {"person_id": f"stub-{event_id}", "is_new": True, "confidence": 0.0}


def estimate_height(record, model="steplength_linear"):
    """record: dict (MATLAB struct) with data, fs, dx, gauge_length, x0, units, data_label."""
    rec = dict(record)
    missing = [k for k in ("data", "fs", "dx", "gauge_length") if k not in rec]
    if missing:
        raise KeyError(f"record is missing fields: {missing}")
    unknown = [k for k in rec if k not in _RECORD_KEYS]
    if unknown:
        raise KeyError(f"record has unexpected fields: {unknown}")
    return _estimate_height(rec["data"], rec["fs"], rec["dx"], rec["gauge_length"],
                            x0=rec.get("x0", 0.0), units=rec.get("units", "strain"),
                            data_label=rec.get("data_label", "SIMULATED"), model=model)


__all__ = ["classify_event", "profile_person", "estimate_height", "OUTPUT_FIELDS"]
