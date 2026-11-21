%% export_roundtrip_test.m
% Writes ONE small test file, roundtrip_test.h5, in the current folder, in
% exactly the format make_height_dataset.m uses, filled with a known
% pattern X(t,c) = 1000*t + c (t, c counted from 0). Then run
%     python tests/check_matlab_roundtrip.py roundtrip_test.h5
% to prove that Python reads MATLAB's layout, units and labels correctly.
% Run it in an empty scratch folder; it touches no other file.
clear; clc;
fn = 'roundtrip_test.h5';
if exist(fn, 'file'), delete(fn); end
Nt = 600;  Nch = 161;
[c, t] = meshgrid(0:Nch-1, 0:Nt-1);          % t down the rows, c across
X = 1000*t + c;                               % [Nt x Nch], as in the generator
h5create(fn, '/data', size(X), 'Datatype', 'single');
h5write(fn, '/data', single(X));
h5writeatt(fn, '/data', 'fs', 500);
h5writeatt(fn, '/data', 'dx', 0.25);
h5writeatt(fn, '/data', 'gauge_length', 1.0);
h5writeatt(fn, '/data', 'x0', 0);
h5writeatt(fn, '/data', 'units', 'strain');
h5writeatt(fn, '/data', 'source', 'sim');
h5writeatt(fn, '/data', 'label_subject_id', 'S01');
h5writeatt(fn, '/data', 'label_pass_id', 'S01_d1.0_r1');
h5writeatt(fn, '/data', 'label_height_cm', 163.4);
h5writeatt(fn, '/data', 'label_true_step_length_m', 0.678);
Y = h5read(fn, '/data');                      % MATLAB reads it back
assert(isequal(size(Y), [Nt Nch]) && Y(8, 4) == 7003, 'MATLAB read-back failed');
fprintf('Wrote %s (%d x %d). MATLAB read-back OK. Now run the Python check.\n', fn, Nt, Nch);
