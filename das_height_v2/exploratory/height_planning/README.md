# exploratory/height_planning/ — Phase B planning model (SIMULATED + ASSUMED)

How much stature information can a speed-compensated gait model recover, and
how many participants does Phase B need to show it? A planning tool, not
evidence. No DAS signal is simulated; DAS enters only through its step-length
error measured in T10 (0.5 %). Background and literature: the "Height from
Gait with DAS" document, sections 8–11.

Model: step length = K × height × cadence × e^u × e^noise (walk ratio constant
within a person and scaling with stature). u ~ N(0, σu) is the person-specific
gait style, the unknown biological limit. Assumed population: 50 % men N(175, 7)
cm, 50 % women N(162, 6.5) cm; self-selected cadence spread 8 % (or 12 %);
40 steps per pass; step noise 1.5 % after speed. Models (leave-one-subject-out):
A step length only; B one self-selected pass + cadence (the field case);
C person mean over six speed conditions (the research curve).

Run: `python -m exploratory.height_planning.run` (about 20 s). Tests:
`tests/test_height_planning.py`.

## Results (seed 2026)

**Large sample (n = 2000):** the published uncompensated r (0.3–0.45) appears
for every σu, so the literature cannot tell the limit; only an experiment that
measures speed can.

| σu | r A (cadence spread 8 % / 12 %) | r B, speed-compensated | RMSE B (cm), baseline ≈ 9.3 |
|---|---|---|---|
| 3 % | 0.54 / 0.42 | 0.87–0.88 | 4.4–4.5 |
| 5 % | 0.51 / 0.36 | 0.73–0.74 | 6.4 |
| 8 % | 0.43 / 0.36 | 0.55–0.57 | 7.6–7.7 |
| 12 % | 0.36 / 0.28 | 0.42 | 8.4–8.5 |

Single pass + cadence (B) is within 0.1–0.5 cm of the six-speed curve (C).

**What Phase B would see** (300 simulated studies; "proven" = one-sided 95 %
lower bound of r above 0.45):

| σu | 12 people | 20 | 30 | 40 |
|---|---|---|---|---|
| 3 % | 69 % | 94 % | 99 % | 100 % |
| 5 % | 20 % | 42 % | 68 % | 76 % |
| 8 % | 3 % | 6 % | 10 % | 15 % |

Reading: 12–15 people is a pilot that measures σu; proving r > 0.45 needs about
20 people if σu ≈ 3 % and 40 or more if σu ≈ 5 %; at 8 % it is not provable at
realistic sizes.
