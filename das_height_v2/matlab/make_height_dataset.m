%% make_height_dataset.m
% -------------------------------------------------------------------------
% Generate SIMULATED walking passes with known heights and export one HDF5
% file per pass for the Python height pipeline (das_height).
%
% Status : SIMULATED ONLY. The height information in these files comes from
%          ONE assumption: step length = k * height. Any accuracy computed
%          on these files only shows that the pipeline recovers that
%          assumption. It is NOT a measure of real-world performance.
%
% - Standalone: does not modify the phi-OTDR simulation or
%   DAS_Profiling_Sim.m. Acquisition parameters can be copied from them.
% - Base MATLAB R2016b or newer. No toolboxes.
%
% Changes compared with DAS_Profiling_Sim.m:
%   1. DAS directivity: axial strain of a radially outgoing ground wave,
%        eps_xx = du_r/dr * cos^2(theta) + (u_r / r) * sin^2(theta)
%      so a footstep beside the fibre gives a weak "broadside" response at
%      the nearest channel and a two-lobed pattern along the fibre.
%   2. Each subject has their own step-length/height ratio k (fixed across
%      passes) plus small pass-to-pass variation, instead of a new random k
%      for every pass.
%   3. Cadence (steps/min) is drawn per pass; speed = step length x cadence.
%   4. Several passes per subject at different lane offsets, as in the
%      planned PNU experiment, exported as HDF5 with labels.
%
% ASSUMPTIONS still to replace with measured or cited values:
%   k mean 0.415, between-person SD 0.02, within-person SD 1 %;
%   cadence 95-125 steps/min, independent of height; ground response
%   25 Hz; wave speed 200 m/s; white noise; straight parallel path.
% -------------------------------------------------------------------------
clear; clc; rng(1);
outDir = 'sim_out';
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% Acquisition (replace with team / A1-NEO values)
P.fs    = 500;     % sampling rate per channel [Hz]
P.dx    = 0.25;    % channel spacing [m]
P.GL    = 1.0;     % gauge length [m]
P.Lfib  = 40;      % test section length [m]
P.noise = 0.05;    % noise relative to a reference footstep (per dx)

%% Ground (site-dependent, unknown until measured)
S.depth = 0.3;  S.f0 = 25;  S.zeta = 0.15;  S.alpha = 0.15;
S.G = 1e-3;     S.vs = 200;

%% Gait (ASSUMPTIONS)
G.kL_mean = 0.415;  G.kL_std = 0.02;  G.kL_pass_std = 0.01;
G.cad_range = [95 125];  G.aF = 1.15;  G.Tc = 0.12;
G.weight_kg = 70;   % constant: weight is NOT studied in this dataset

%% Experiment design (mirrors the planned PNU protocol)
nSubj   = 30;
H_range = [1.50 1.80];       % [m]; widen it if your volunteers allow
offsets = [0.5 1.0 2.0];     % walking lanes, distance from fibre [m]
nRep    = 2;                 % passes per lane

rows = {};
for s = 1:nSubj
    H   = H_range(1) + diff(H_range)*rand;
    k   = G.kL_mean + G.kL_std*randn;        % this subject's own ratio
    sid = sprintf('S%02d', s);
    for d = offsets
        for r = 1:nRep
            [X, tr] = simulate_pass(H, k, d, P, S, G);
            pid = sprintf('%s_d%.1f_r%d', sid, d, r);
            lab = struct('subject_id', sid, 'pass_id', pid, ...
                         'height_cm', 100*H, 'lane_offset_m', d, ...
                         'true_step_length_m', tr.L);
            write_pass_h5(fullfile(outDir, [pid '.h5']), X, P, lab);
            rows(end+1,:) = {pid, sid, 100*H, d, tr.L, tr.cad}; %#ok<AGROW>
        end
    end
    fprintf('Subject %2d/%d done\n', s, nSubj);
end
T = cell2table(rows, 'VariableNames', {'pass_id','subject_id','height_cm', ...
    'lane_offset_m','true_step_length_m','true_cadence_spm'});
writetable(T, fullfile(outDir, 'manifest_truth.csv'));
fprintf('Wrote %d passes to %s\n', height(T), outDir);

%% ======================= LOCAL FUNCTIONS ===============================

function [X, tr] = simulate_pass(H, k, d, P, S, G)
xc  = 0:P.dx:P.Lfib;  Nch = numel(xc);
L   = k*H*(1 + G.kL_pass_std*randn);          % step length [m]
cad = G.cad_range(1) + diff(G.cad_range)*rand; % steps per minute
T   = 60/cad;                                  % step period [s]
xs  = 3:L:(P.Lfib-3);  Ns = numel(xs);
ts  = 1.0 + (0:Ns-1)*T + 0.01*randn(1,Ns);
dk  = d + 0.08*(-1).^(1:Ns);                   % left / right foot
F   = G.aF*G.weight_kg*9.81*(1 + 0.05*randn(1,Ns));
Nt  = round((ts(end) + 1.0)*P.fs);
X   = zeros(Nt, Nch);
[w, dw] = footstep_waveform(P.fs, S, G);  Nw = numel(w);
for j = 1:Ns
    rh2  = (xc - xs(j)).^2 + dk(j)^2;          % horizontal distance^2
    r    = sqrt(rh2 + S.depth^2);
    cos2 = (xc - xs(j)).^2 ./ rh2;  sin2 = 1 - cos2;
    A    = S.G*F(j).*exp(-S.alpha*r);
    n0   = round((ts(j) + r/S.vs)*P.fs) + 1;
    for c = 1:Nch
        idx = n0(c):min(n0(c)+Nw-1, Nt);  m = numel(idx);
        X(idx,c) = X(idx,c) + A(c)*(-dw(1:m)/S.vs*cos2(c) + w(1:m)/r(c)*sin2(c));
    end
end
Aref = S.G*G.aF*75*9.81;
X = X + P.noise*Aref*sqrt(P.GL/P.dx)*randn(Nt, Nch);   % noise per segment
nGL = max(1, round(P.GL/P.dx));
if nGL > 1, X = conv2(X, ones(1,nGL)/nGL, 'same'); end % gauge-length average
tr = struct('L', L, 'cad', cad, 'xs', xs, 'ts', ts);
end

function [w, dw] = footstep_waveform(fs, S, G)
t  = (0:round(0.4*fs)-1).'/fs;
p  = sin(pi*t/G.Tc) .* (t <= G.Tc);
wd = 2*pi*S.f0*sqrt(1 - S.zeta^2);
h  = exp(-S.zeta*2*pi*S.f0*t) .* sin(wd*t);
w  = conv(p, h);  w = w(1:numel(t));  w = w/max(abs(w));
dw = gradient(w)*fs;                           % time derivative
end

function write_pass_h5(fn, X, P, lab)
% One pass per file. MATLAB writes column-major, so Python (h5py) sees the
% transpose; das_height/io_sim.py transposes it back.
if exist(fn, 'file'), delete(fn); end
h5create(fn, '/data', size(X), 'Datatype', 'single');
h5write(fn, '/data', single(X));
h5writeatt(fn, '/data', 'fs', P.fs);
h5writeatt(fn, '/data', 'dx', P.dx);
h5writeatt(fn, '/data', 'gauge_length', P.GL);
h5writeatt(fn, '/data', 'x0', 0);
h5writeatt(fn, '/data', 'units', 'strain');
h5writeatt(fn, '/data', 'source', 'sim');
f = fieldnames(lab);
for i = 1:numel(f)
    h5writeatt(fn, '/data', ['label_' f{i}], lab.(f{i}));
end
end
