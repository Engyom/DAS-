# T11 — Load-change detection and weight-range identifiability (EXPLORATORY)

**Status: SIMULATED + ASSUMED. No weight model was trained. Nothing here shows
that DAS can estimate body weight.** In the generator, footstep amplitude is
proportional to weight by construction (F = 1.15·W·g); this study takes that
as the *best case* and asks how large every *other* amplitude change is.

Run: `python -m exploratory.load_identifiability.run` (from `das_height_v2/`, ~45 s).
Setting: channel spacing 0.4 m; gauge length 0.8 m and 1.2 m (whole multiples of
the spacing, so MATLAB and Python average the same channels); 8 simulated passes
per point; amplitude = ½·ln(median footstep energy) from the pipeline's own
`detect_footsteps`.

## Method

ln A = ln W + ln(lane) + ln(coupling) + ln(speed) + ln(footwear) + noise.
Two weights are separable when d′ = ln(W₂/W₁)/σ_total ≥ 2 (≈ 16 % error between
two classes). Classes across 45–110 kg are spaced so neighbours reach d′ = 2.

| Term | Source | Value used |
|---|---|---|
| Repeatability (noise, 5 % step-to-step force, channel grid) | [Sim] | σ = 0.019 / 0.017 per pass (GL 0.8 / 1.2 m); 0.13 / 0.11 per footstep |
| Lane distance | [Sim] | −0.94 / −0.81 ln per m at 1 m (≈ −9 % per 0.1 m); 0.5 → 2 m: amplitude ÷ 3.4 / ÷ 2.9 |
| Lane unknown in 0.5–2 m | [Sim] | σ = 0.36 / 0.32 |
| Fibre–ground coupling (incl. channel sensitivity) | **[Assumption]**, swept | 0.40 uncalibrated; 0.10 calibrated |
| Walking speed → force | **[Assumption]**, swept (direction of effect is established; size not yet cited, task T6) | 0.15 free speed; 0.05 metronome; 0.03 same speed |
| Footwear | **[Assumption]**, swept | 0.15 own shoes; 0.10 controlled; 0 same shoes |
| Check: slope of ln A on ln W | [Sim] | 0.98 / 0.99 = the generator's built-in proportionality, not evidence |

## Results (GL 0.8 m; GL 1.2 m within ±0.03 in σ)

| Calibration level | σ_total | Min. weight ratio for d′ = 2 | Weight classes 45–110 kg | Error 60 vs 85 kg |
|---|---|---|---|---|
| L0 uncalibrated (lane unknown 0.5–2 m, free speed, own shoes) | 0.58 | ×3.18 | **1** (no discrimination) | 38 % (chance = 50 %) |
| L1 lane marked ±0.25 m, coupling not calibrated | 0.51 | ×2.77 | **1** | 37 % |
| L2 lane ±0.1 m, coupling calibrated to 10 %, metronome speed | 0.18 | ×1.43 | **3** (e.g. ~50 / ~72 / ~103 kg) | 16 % |
| L3 same person, place, shoes and speed (backpack design) | 0.06 | ×1.12 (within one person) | not applicable | — |

Figures: `results/amplitude_confounds.png` (lane, weight and crossing curves),
`results/confound_budget.png` (variance contributions per level).

**Crossing intruders.** For one footstep along a path across the fibre, amplitude
changes by ×7 (GL 0.8) / ×5 (GL 1.2) between 0.25 and 3 m lateral distance, while
a single fibre does not measure that distance. A crossing gives only a few
footsteps, each at a different, unknown distance: **no weight information can be
separated from distance for crossings.** (A footstep exactly on the fibre is a
singular point of the simulation's geometry and is excluded.)

**Backpack trial planning** (within one person; passes per load level for 80 %
power, two-sided α = 0.05):

| Body | Load | Expected amplitude change [Sim] | σ_pass = 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|---|
| 70 kg | +10 kg | +14 % | 3 | 9 | 20 |
| 70 kg | +20 kg | +29 % | 1 | 3 | 6 |
| 85 kg | +10 kg | +12 % | 4 | 13 | 29 |

The simulated pass-to-pass σ (≈ 0.02) would need only 1 pass; real repeatability
will be worse and must be measured in Phase A (go/no-go criterion 4).

## Interpretation

1. **Between people, without calibration, weight ranges are not identifiable**
   (L0/L1: one class). Lane distance and coupling each exceed the whole
   45–110 kg weight effect (ln 2.4 = 0.89 vs lane drop 1.23).
2. **With strong calibration (L2), at most about three broad weight classes**
   become separable, and only if the assumed coupling, speed and footwear
   spreads are right. The dominant requirements: lane known to ±0.1 m and
   coupling calibrated to about 10 %.
3. **Within one person (L3), a load change of roughly +12 % amplitude
   (≈ +8–10 kg) is detectable in this best case.** This supports keeping the
   optional backpack trial as the only real-data weight-related test.
4. The proposed 15 kg-wide bins (40–55, 55–70, 70–85, 85–100 kg) need ratios
   of 1.2–1.4, so they are not separable at any level except a within-person
   design.

## Limits of this study

- Amplitude ∝ weight is an assumption of the generator; real footstep force
  is only approximately proportional to weight and depends on gait and speed.
- The waveform shape is identical for every weight, so temporal or spectral
  weight features cannot be studied with this generator.
- Coupling, speed and footwear sizes are assumptions; φ-OTDR fading and real
  noise are not modelled. Real repeatability is unknown until Phase A.
- Nothing here is a weight estimate, and no result may be described as one.
