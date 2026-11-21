"""DAS-based walker height estimation pipeline.

Status: research prototype. Tested on SIMULATED data only.
No result produced by this package on simulated data is a measure of
real-world height-estimation accuracy.
"""
from .schema import DASRecord
from .footsteps import detect_footsteps
from .features import extract_pass_features, FEATURE_NAMES
