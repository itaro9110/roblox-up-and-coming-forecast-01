# Roblox Up-and-Coming Forecaster

make.fun competition entry: `roblox-up-and-coming-forecast-01`.

Forecasts the Up-and-Coming chart top 10 for each hour of the 72h after
entries close. `main.py` (Python 3.13, stdlib only) reads
`/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json`.

Model: rank-diffusion Monte Carlo. The chart is extremely volatile (top 10
changes in ~95% of consecutive reads), so the model simulates each candidate
game's rank as a random walk with an empirically estimated transition kernel
(Laplace noise, scale by rank band) plus a small age drift (old games fade),
and reports the time-averaged occupancy of each game at each place 1..10.
No single game ever gets a probability near 1 — spreading mass is what wins
log score.

Backtest (practice window, kit history 2026-10-30 -> 2026-11-07, scored
2026-11-08 -> 2026-11-10): **-34.96** vs the persistence baseline's
**-43.51** (same window, same scorer; the scorer reproduces the baseline's
published practice score exactly). Walk-forward over the whole kit history
(182 test reads): **-36.93** vs -39.79 for the no-drift variant; the earlier
split scores -39.04.

No network calls, no model calls, deterministic (fixed seed).
