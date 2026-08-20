# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Price the basket at a quantile of the total, not as a total of quantiles.

The safety pass re-prices every purchase at its own 0.9 cost quantile and sheds
until that sum fits. On premium that sum reaches 3.85x of a 4.00x cap while the
basket actually costs 2.76x -- the router believes it is nearly out of money with
a third of the budget unspent.

That gap is not a calibration error in the per-episode model, it is the wrong
object. Adding 673 independent upper quantiles gives roughly the worst case of
673 draws, whereas what the cap constrains is one number: the total. Errors
average, so the upper quantile of a sum grows like sqrt(N), not N:

    bound = sum(q50_i) + z * sqrt(sum(sigma_i^2)),   sigma_i ~ (q90_i - q50_i)/1.2816

The assumption that buys the sqrt is independence across episodes, and it is not
free -- a systematically miscalibrated cost model moves every term the same way,
which no amount of averaging removes. So z is swept rather than assumed, and each
setting is judged by the bootstrap overrun rate, not by how much it spends.

Baseline to beat: dev weighted 0.6721, P(over) 0% on all three tiers.
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

import exp_gain as E

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
TIERS = list(POL["tiers"])
Z90 = 1.2815515655446004


def route_sum(rows, gain, cost, light_hat, tier, z, slack=E.SLACK):
    """Fill on the median, then bound the total and shed until the bound fits."""
    n = len(rows)
    est = float(light_hat.sum())
    target = float(POL["tiers"][tier]["budget_multiplier"]) * est * (1.0 - slack)

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

    def bound(sel):
        if not sel:
            return est
        mid = sum(cost[0.5][m][i] for _, i, m in sel)
        var = sum(((cost[0.9][m][i] - cost[0.5][m][i]) / Z90) ** 2 for _, i, m in sel)
        return est + mid + z * float(np.sqrt(var))

    while picks and bound(picks) > target:
        picks.pop()

    choice = [L] * n
    for _, i, m in picks:
        choice[i] = m
    tl = sum(r["cost"][L] for r in rows)
    tc = sum(rows[i]["cost"][choice[i]] for i in range(n))
    cap = float(POL["tiers"][tier]["budget_multiplier"]) * tl
    s = sum(rows[i]["score"][choice[i]] for i in range(n)) / n
    return (s if tc <= cap else 0.0), tc / tl, tc <= cap


def evaluate(label, dev, gain, cost, light_hat, router, idxs):
    w = radj = 0.0
    cells = []
    for tier in TIERS:
        s, sp, _ = router(dev, gain, cost, light_hat, tier)
        ratios = np.array([router([dev[i] for i in ix],
                                  {m: gain[m][ix] for m in UP},
                                  {q: {m: cost[q][m][ix] for m in UP} for q in cost},
                                  light_hat[ix], tier)[1] for ix in idxs])
        cap = float(POL["tiers"][tier]["budget_multiplier"])
        p_over = float(np.mean(ratios > cap))
        wt = float(POL["tiers"][tier]["weight"])
        w += wt * s
        radj += wt * s * (1 - p_over)
        cells.append(f"{tier[:4]} {s:.4f} {sp:.2f}/{cap:.2f} {p_over:4.0%}")
    print(f"  {label:28s} {w:8.4f} {radj:8.4f}   " + "  ".join(cells))
    return w, radj


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    Xa, Xb = E.vectorise([r["text"] for r in train], [r["text"] for r in dev])
    gain = E.gain_A(Xa, Xb, train)
    cost, light_hat = E.cost_models(train, dev)
    rng = np.random.default_rng(20260820)
    idxs = [rng.integers(0, len(dev), len(dev)) for _ in range(200)]

    print(f"  {'design':28s} {'weighted':>8} {'risk-adj':>8}   per tier: score spend/cap P(over)")
    evaluate("current: sum of quantiles", dev, gain, cost, light_hat, E.route, idxs)
    for z in (2.0, 3.0, 4.0, 6.0, 8.0):
        evaluate(f"quantile of sum, z={z}", dev, gain, cost, light_hat,
                 lambda r, g, c, l, t, z=z: route_sum(r, g, c, l, t, z), idxs)
