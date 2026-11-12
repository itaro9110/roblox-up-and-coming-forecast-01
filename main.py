#!/usr/bin/env python3
"""Roblox Up-and-Coming forecaster (make.fun roblox-up-and-coming-forecast-01).

Model: rank-diffusion Monte Carlo.
  * Read every history snapshot (up-and-coming chart, rank order + playing).
  * Estimate an empirical rank-transition kernel: Laplace noise whose scale
    (mean |rank change| per sqrt(hour)) depends on the rank band. Top ranks
    move slowly, mid ranks move fast.
  * Each candidate game (ranks 1..40 in the last snapshot) is simulated as a
    random walk for 72h in 3h steps, with a small age drift (old games fade,
    young games hold) and Laplace noise from the kernel. 1500 simulations.
  * The time-averaged occupancy of each game at each place 1..10 is the
    forecast probability; a temperature and an unnamed-mass reserve (for games
    that appear during the window and were never seen) are applied.
  * top10: one game per place, chosen greedily by tempered probability;
    any unfilled place takes the remaining game with the lowest mean
    simulated rank.

Design notes (validated on the practice window, kit history 2026-10-30 ->
2026-11-07, scored 2026-11-08 -> 2026-11-10):
  * Persistence baseline scores -43.51 on the practice window; this model
    scores -34.96 there at 1500 sims (same window, same scorer; -35.42 at
    600 sims — the difference is Monte Carlo noise). Walk-forward over the
    whole kit history (182 test reads): -36.93 vs -39.79 for the no-drift
    variant; the earlier split scores -39.04.
  * The chart is extremely volatile: the top 10 changes in ~95% of
    consecutive reads, so no single game ever gets a probability near 1 at
    any place. Spreading mass is what wins log score; a miss at p=0.0001
    costs -9.21.
  * Signals tested and REJECTED (they help one window and hurt another):
    log-playing momentum, rank-velocity drift, jump follow-through, and
    history-fitted occupancy. The age drift is small and kept because it is
    the only drift that helps on every validation split.
  * Reads only /input/history.ndjson + /input/task.json. No network, no
    model calls, no file writes outside /output. Deterministic (fixed seed):
    the same input always produces the same forecast.

Stdlib only. Python 3.13.
"""
import json
import math
import os
import random
from datetime import datetime

# --------------------------------------------------------------------------
# Tunable constants (selected by walk-forward validation over the kit
# history + practice window; see BUILD_LOG.md).
N_SIMS = 1500          # Monte Carlo simulations
STEP_H = 3.0           # simulation step, hours
HORIZON_H = 72.0       # forecast horizon, hours
POOL_TOP = 40          # candidate games: last-snapshot ranks 1..40
NOISE_SCALE = 1.0      # multiplier on the empirical kernel scale
AGE_A = 0.35           # age drift: ranks/step ~ (AGE_A*age_days + AGE_B)/7
AGE_B = -0.1           #   (age = days since the game first appeared)
TEMPERATURE = 1.0      # tempering exponent on place probabilities
UNNAMED = 0.2          # mass reserved per place for games not in the pool
SEED = 12345
FLOOR = 0.0001         # scorer floor; probabilities are never written below it

# Empirical kernel: mean |rank change| per sqrt(hour), by starting rank band.
# Estimated from all consecutive-read transitions in the input history.
# Fallback values (from the kit history) are used if the input has too few
# transitions to estimate a band.
KERNEL_FALLBACK = {(1, 3): 0.277, (4, 10): 0.616, (11, 20): 1.116,
                   (21, 40): 1.7, (41, 60): 2.22}
BANDS = [(1, 3), (4, 10), (11, 20), (21, 40), (41, 60)]


def parse_ts(s):
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def load_history(input_dir):
    """Return a list of snapshots: {'at': datetime, 'chart': [uid,...],
    'playing': {uid: int}} sorted by time. Handles the kit format
    (history.ndjson rows with 'up_and_coming' lists) and tolerates rows
    that are missing fields or whole charts."""
    path = os.path.join(input_dir, "history.ndjson")
    snaps = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                try:
                    at = parse_ts(row.get("at") or row.get("generated_at") or row.get("ts"))
                except Exception:
                    continue
                chart = []
                playing = {}
                rows = row.get("up_and_coming") or row.get("chart") or []
                for r in rows:
                    if isinstance(r, dict):
                        uid = r.get("universe_id")
                        if uid is None:
                            continue
                        chart.append((int(r.get("rank") or len(chart) + 1), int(uid)))
                        playing[int(uid)] = int(r.get("playing") or 0)
                    elif isinstance(r, (int, str)):
                        chart.append((len(chart) + 1, int(r)))
                chart.sort()
                snaps.append({"at": at, "chart": [u for _, u in chart],
                              "playing": playing})
    except FileNotFoundError:
        return []
    snaps.sort(key=lambda s: s["at"])
    return snaps


def band_of(rank):
    for lo, hi in BANDS:
        if lo <= rank <= hi:
            return (lo, hi)
    return BANDS[-1]


def transition_kernel(snaps):
    """Mean |dr| / sqrt(hours) per rank band, from consecutive snapshots."""
    vals = {b: [] for b in BANDS}
    for s1, s2 in zip(snaps, snaps[1:]):
        dt = (s2["at"] - s1["at"]).total_seconds() / 3600.0
        if dt <= 0:
            continue
        pos1 = {u: i + 1 for i, u in enumerate(s1["chart"][:60])}
        pos2 = {u: i + 1 for i, u in enumerate(s2["chart"][:60])}
        for u, r1 in pos1.items():
            if u in pos2:
                vals[band_of(r1)].append(abs(pos2[u] - r1) / math.sqrt(dt))
    out = {}
    for b in BANDS:
        v = vals[b]
        out[b] = (sum(v) / len(v)) if len(v) >= 5 else KERNEL_FALLBACK[b]
    return out


def features(snaps):
    """Per game: last rank, age (days since first seen), last rank change."""
    games = {}
    for s in snaps:
        for i, u in enumerate(s["chart"][:60]):
            g = games.setdefault(u, {"ranks": []})
            g["ranks"].append((s["at"], i + 1))
    last_at = snaps[-1]["at"]
    feats = {}
    for u, g in games.items():
        ranks = g["ranks"]
        last_dr = ranks[-1][1] - ranks[-2][1] if len(ranks) >= 2 else 0
        feats[u] = {
            "r0": ranks[-1][1],
            "age": (last_at - ranks[0][0]).total_seconds() / 86400.0,
            "last_dr": last_dr,
            "last_seen": ranks[-1][0],
        }
    return feats, last_at


def simulate(feats, kernel, last_chart):
    """Monte Carlo over the candidate pool. Returns (place_dist, mean_rank):
    place_dist[place][uid] = fraction of (sim, step) with uid at that place;
    mean_rank[uid] = time-averaged rank over all sims and steps."""
    # Pool = ONLY the last snapshot's top POOL_TOP games. Using any game
    # whose last-seen rank was <= POOL_TOP would admit stale games that have
    # already left the chart and steal occupancy mass.
    pool = [u for u in last_chart[:POOL_TOP] if u in feats]
    rng = random.Random(SEED)
    n_steps = int(HORIZON_H / STEP_H)
    place_dist = {k: {} for k in range(1, 11)}
    rank_sum = {u: 0.0 for u in pool}
    rank_cnt = {u: 0 for u in pool}
    for _ in range(N_SIMS):
        ranks = {u: float(feats[u]["r0"]) for u in pool}
        for step in range(n_steps):
            order = sorted(pool, key=lambda u: ranks[u])
            for pos, u in enumerate(order[:10], start=1):
                d = place_dist[pos]
                d[u] = d.get(u, 0.0) + 1.0
            for u in pool:
                rank_sum[u] += ranks[u]
                rank_cnt[u] += 1
            for u in pool:
                scale = (kernel[band_of(max(1, int(round(ranks[u]))))]
                         * NOISE_SCALE * math.sqrt(STEP_H))
                lap = rng.expovariate(1.0 / scale) - rng.expovariate(1.0 / scale) \
                    if scale > 0 else 0.0
                age_now = feats[u]["age"] + (step * STEP_H) / 24.0
                dr = (AGE_A * age_now + AGE_B) * (STEP_H / 7.0)
                ranks[u] = max(1.0, ranks[u] + dr + lap)
    total = N_SIMS * n_steps
    for k in range(1, 11):
        for u in place_dist[k]:
            place_dist[k][u] /= total
    mean_rank = {u: (rank_sum[u] / rank_cnt[u]) for u in pool}
    return place_dist, mean_rank


def finalize(place_dist, mean_rank):
    pd = {}
    for k in range(1, 11):
        d = place_dist.get(k, {})
        if not d:
            pd[k] = {}
            continue
        tp = {u: p ** TEMPERATURE for u, p in d.items()}
        s = sum(tp.values())
        pd[k] = {u: max((p / s) * (1.0 - UNNAMED), FLOOR) for u, p in tp.items()}
    # top10: greedy by tempered probability per place, then fill by mean rank
    pairs = sorted(((p, k, u) for k in range(1, 11) for u, p in pd[k].items()),
                   reverse=True)
    top10 = [None] * 10
    used = set()
    for p, k, u in pairs:
        if top10[k - 1] is None and u not in used:
            top10[k - 1] = u
            used.add(u)
    for i in range(10):
        if top10[i] is None:
            rest = sorted((u for u in mean_rank if u not in used),
                          key=lambda u: mean_rank[u])
            if rest:
                top10[i] = rest[0]
                used.add(rest[0])
    top10 = [u for u in top10 if u is not None]
    probs = []
    for k in range(1, 11):
        for u, p in sorted(pd[k].items()):
            if p > 0.0005:
                probs.append({"universe_id": int(u), "place": k,
                              "p": round(float(p), 6)})
    return {"probabilities": probs, "top10": [int(u) for u in top10]}


def fallback_forecast(snaps):
    """If the history is unusable (empty, one snapshot, no pool), fall back
    to persistence on the last chart with an honest spread, or to an empty
    forecast that still satisfies the schema."""
    if not snaps or not snaps[-1]["chart"]:
        return {"probabilities": [], "top10": []}
    chart = snaps[-1]["chart"][:10]
    probs = []
    for k, u in enumerate(chart, start=1):
        # persistence mass at the same place + a decaying tail over the
        # other top-10 games; the rest stays unnamed.
        for j, v in enumerate(chart, start=1):
            p = 0.55 if j == k else 0.05
            probs.append({"universe_id": int(v), "place": k, "p": round(p, 6)})
    return {"probabilities": probs, "top10": [int(u) for u in chart]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    snaps = load_history(input_dir)
    if len(snaps) < 2 or not snaps[-1]["chart"]:
        forecast = fallback_forecast(snaps)
    else:
        kernel = transition_kernel(snaps)
        feats, _ = features(snaps)
        pool = [u for u in snaps[-1]["chart"][:POOL_TOP] if u in feats]
        if len(pool) < 10:
            forecast = fallback_forecast(snaps)
        else:
            place_dist, mean_rank = simulate(feats, kernel, snaps[-1]["chart"])
            forecast = finalize(place_dist, mean_rank)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w", encoding="utf-8") as fh:
        json.dump(forecast, fh)


if __name__ == "__main__":
    main()
