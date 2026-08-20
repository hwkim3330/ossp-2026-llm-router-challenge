# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Is one safety quantile for all three tiers leaving score on the table?

The tiers are not symmetric. Fast runs at 1.11x of a 1.25x cap and premium at
2.76x of 4.00x, so the same quantile buys very different amounts of headroom, and
the public forks that are furthest along both set the safety level per tier --
one of them explicitly promoting an "EV-optimal premium q068".

That is the shape of the intervention that scored 0.4779 here, below routing
everything to light, so it is worth measuring rather than assuming either way.
Every combination of q_safe in {0.8, 0.9, 0.95} per tier is scored on dev and by
200 bootstrap resamples, and reported as expected score net of the chance of
forfeiting the tier:

    risk-adjusted = sum over tiers of weight * score * (1 - P(over cap))

A per-tier setting is worth taking only if it beats the uniform one on that
number, not on the dev score it advertises.
"""
import importlib.util, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location(
    "rd", str(Path(__file__).resolve().parent / "router_dev.py"))
rd = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(rd)

from sklearn.ensemble import HistGradientBoostingRegressor

import exp_gain as E

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
TIERS = list(POL["tiers"])
GRID = (0.8, 0.9, 0.95)


def cost_models(tr, te, qs):
    Fa, Fb = E.cost_features(tr), E.cost_features(te)
    cost = {q: {} for q in qs}
    for m in UP:
        yc = np.clip([r["cost"][m] - r["cost"][L] for r in tr], 1.0, None)
        for q in qs:
            cost[q][m] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
                max_depth=6, random_state=0).fit(Fa, yc).predict(Fb).clip(1.0)
    yl = np.clip([r["cost"][L] for r in tr], 1.0, None)
    light = HistGradientBoostingRegressor(
        loss="quantile", quantile=E.Q_LIGHT, max_iter=250, learning_rate=0.06,
        max_depth=6, random_state=0).fit(Fa, yl).predict(Fb).clip(1.0)
    return cost, light


def route(rows, gain, cost, light_hat, tier, q_safe):
    n = len(rows)
    est = float(light_hat.sum())
    target = float(POL["tiers"][tier]["budget_multiplier"]) * est * (1.0 - E.SLACK)
    picks, spent, taken = [], est, set()
    for eff, i, m in sorted(((gain[m][i] / cost[0.5][m][i], i, m)
                             for i in range(n) for m in UP if gain[m][i] > 0),
                            reverse=True):
        if i in taken:
            continue
        if spent + cost[0.5][m][i] <= target:
            spent += cost[0.5][m][i]
            picks.append((eff, i, m))
            taken.add(i)
    while picks and est + sum(cost[q_safe][m][i] for _, i, m in picks) > target:
        picks.pop()
    choice = [L] * n
    for _, i, m in picks:
        choice[i] = m
    tl = sum(r["cost"][L] for r in rows)
    tc = sum(rows[i]["cost"][choice[i]] for i in range(n))
    cap = float(POL["tiers"][tier]["budget_multiplier"]) * tl
    s = sum(rows[i]["score"][choice[i]] for i in range(n)) / n
    return (s if tc <= cap else 0.0), tc / tl


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    Xa, Xb = E.vectorise([r["text"] for r in train], [r["text"] for r in dev])
    gain = E.gain_A(Xa, Xb, train)
    qs = sorted({0.5, *GRID})
    cost, light_hat = cost_models(train, dev, qs)
    rng = np.random.default_rng(20260820)
    idxs = [rng.integers(0, len(dev), len(dev)) for _ in range(200)]

    # each tier is independent given the cost model, so score them once each
    print(f"  {'tier':9s} {'q_safe':>6} {'score':>7} {'spend/cap':>11} {'P(over)':>8} "
          f"{'w*score':>8} {'w*score*(1-p)':>14}")
    best = {}
    for tier in TIERS:
        cap = float(POL["tiers"][tier]["budget_multiplier"])
        w = float(POL["tiers"][tier]["weight"])
        for q in GRID:
            s, sp = route(dev, gain, cost, light_hat, tier, q)
            ratios = np.array([route([dev[i] for i in ix],
                                     {m: gain[m][ix] for m in UP},
                                     {qq: {m: cost[qq][m][ix] for m in UP} for qq in cost},
                                     light_hat[ix], tier, q)[1] for ix in idxs])
            p = float(np.mean(ratios > cap))
            radj = w * s * (1 - p)
            print(f"  {tier:9s} {q:6} {s:7.4f} {sp:6.2f}/{cap:.2f} {p:8.1%} "
                  f"{w * s:8.4f} {radj:14.4f}")
            if tier not in best or radj > best[tier][0]:
                best[tier] = (radj, q, w * s)
        print()
    print(f"  uniform q_safe=0.90 risk-adjusted "
          f"{sum(w for w, q, _ in [(v[0], v[1], v[2]) for v in best.values()]) if False else ''}")
    print("  best per tier: " + "  ".join(f"{t}={v[1]}" for t, v in best.items()))
    print(f"  per-tier risk-adjusted total {sum(v[0] for v in best.values()):.4f}")
