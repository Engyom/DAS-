% matlab_example.m  -  calling estimate_height from MATLAB (T12)
%
% STATUS: TESTED 22 Nov 2026 on MATLAB R2026b (Windows 11) with Python 3.14.10
% from the repository's .venv, OutOfProcess. Output for S01_d1.0_r1.h5:
%   size 12368 x 161; height 168.0 cm (+/- 7.2), stride 1.424 m,
%   cadence 124.0 spm, QC: ok   (identical to the Python side).
% SIMULATED model: this checks the MATLAB-Python link, not real-world accuracy.
%
% Requirements: Python with numpy, scipy, pandas, scikit-learn
% (das_height_v2/requirements.txt), and this repository on disk.

% 1) Python environment: once per MATLAB session, BEFORE any py.* call
pe = pyenv;
if pe.Status == "NotLoaded"
    % Either a version string ("3.13") or the full path to a Python executable,
    % e.g. the project's .venv: "...\das_height_v2\.venv\Scripts\python.exe"
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
f = "S01_d1.0_r1.h5";                      % one file from sim_out/
X = double(h5read(f, "/data"));            % MATLAB reads back its own layout: time x channel, no transpose
assert(size(X,1) > size(X,2), "expected time x channel");
record = struct( ...
    "data",         py.numpy.array(X), ...
    "fs",           double(h5readatt(f, "/data", "fs")), ...
    "dx",           double(h5readatt(f, "/data", "dx")), ...
    "gauge_length", double(h5readatt(f, "/data", "gauge_length")), ...
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
