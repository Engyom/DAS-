% matlab_example.m  -  calling estimate_height from MATLAB (T12)
%
% STATUS: UNTESTED IN MATLAB. Written and checked against the Python side
% only (tests/test_integration.py). Run it once on a team PC and report.
%
% Requirements: Python 3.13 with numpy, scipy, pandas, scikit-learn
% (das_height_v2/requirements.txt), and this repository on disk.

% 1) Python environment: once per MATLAB session, BEFORE any py.* call
pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv(Version="3.13", ExecutionMode="OutOfProcess");
end

% 2) Make the package importable: the folder that CONTAINS das_height/ and integration/
repo = fullfile(fileparts(mfilename("fullpath")), "..");   % das_height_v2/
if count(py.sys.path, repo) == 0
    insert(py.sys.path, int32(0), repo);
end
bridge = py.importlib.import_module("integration.bridge");

% 3) One walking pass: X is (time x channel), exactly as the simulation stores it.
%    Do NOT transpose. Here: load one simulated pass as an example.
X = double(h5read("S01_d1.0_r1.h5", "/data"));   % MATLAB reads back its own layout: time x channel, no transpose
assert(size(X,1) > size(X,2), "expected time x channel");
record = struct( ...
    "data",         py.numpy.array(X), ...
    "fs",           500, ...
    "dx",           0.25, ...
    "gauge_length", 1.0, ...
    "x0",           0, ...
    "units",        "strain", ...
    "data_label",   "SIMULATED");

% 4) Call and convert the result to a MATLAB struct
out = struct(bridge.estimate_height(record));
height_cm = double(out.height_cm);          % NaN when QC fails
qc_pass   = logical(out.qc_pass);
reason    = string(out.qc_reason);
fprintf("height %.1f cm (+/- %.1f, SIMULATED model), stride %.3f m, cadence %.1f spm, QC: %s\n", ...
    height_cm, double(out.uncertainty_cm), double(out.stride_m), double(out.cadence_spm), reason);
