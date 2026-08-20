# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Redo the quantile-of-a-sum bound with a sigma that is measured, not assumed.

`exp_sum_quantile.py` replaced the sum of per-episode 0.9 quantiles with
`sum(q50) + z*sqrt(sum(sigma^2))` and blew every tier, and I recorded that as
evidence of correlated errors. That was wrong. Resampling residuals shows the sd
of their sum tracks sqrt(N) exactly (ratio 0.99-1.02 at N = 50, 200, 600), so
they are independent and the sqrt is legitimate.

What was wrong is sigma. It came from `(q90 - q50)/1.2816`, which is the normal
relation between an interquantile gap and a standard deviation, and these
residuals are nowhere near normal -- skew 15.2 and kurtosis 463 for `ax31`, 8.3
and 94 for `axk1-think`. The gap between two quantiles of a heavy-tailed
distribution says almost nothing about its spread, and here it understated the
true sd by 3.07x and 2.01x.

So the bound is kept and sigma is rescaled by a factor measured out-of-fold on
train, never on dev:

    sigma_i = k_m * (q90_i - q50_i) / 1.2816,
    k_m     = sd(out-of-fold residuals) / mean((q90 - q50)/1.2816)

The per-episode shape still comes from the model, so heteroscedasticity survives;
only the scale is corrected. With N in the hundreds the sum is close to normal
even at this kurtosis (kurtosis of a mean falls as 3 + (k-3)/N), which is what
makes a z-based bound defensible on the total when it is indefensible per episode.

Judged, as everything else here, by the bootstrap overrun rate rather than spend.
Baseline: dev weighted 0.6721, P(over) 0% everywhere.
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
from sklearn.model_selection import KFold

import exp_gain as E
from exp_sum_quantile import Z90

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
TIERS = list(POL["tiers"])


def sigma_scale(train, folds=4):
    """k_m from out-of-fold residuals on train. Using dev here would be cheating
    the very quantity the bound is supposed to be honest about."""
    rows = np.array(train, dtype=object)
    kf = KFold(n_splits=folds, shuffle=True, random_state=20260820)
    res = {m: [] for m in UP}
    gap = {m: [] for m in UP}
    for tr_i, te_i in kf.split(rows):
        tr, te = list(rows[tr_i]), list(rows[te_i])
        cost, _ = E.cost_models(tr, te)
        for m in UP:
            true = np.array([r["cost"][m] - r["cost"][L] for r in te])
            res[m].append(true - cost[0.5][m])
            gap[m].append((cost[0.9][m] - cost[0.5][m]) / Z90)
    out = {}
    for m in UP:
        r = np.concatenate(res[m])
        g = np.concatenate(gap[m])
        out[m] = float(r.std() / max(g.mean(), 1e-9))
    return out


def route_sigma(rows, gain, cost, light_hat, tier, z, k, slack=E.SLACK):
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
        var = sum((k[m] * (cost[0.9][m][i] - cost[0.5][m][i]) / Z90) ** 2
                  for _, i, m in sel)
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


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    k = sigma_scale(train)
    print("sigma scale measured out-of-fold on train: " +
          "  ".join(f"{m} x{v:.2f}" for m, v in k.items()) + "\n")

    Xa, Xb = E.vectorise([r["text"] for r in train], [r["text"] for r in dev])
    gain = E.gain_A(Xa, Xb, train)
    cost, light_hat = E.cost_models(train, dev)
    rng = np.random.default_rng(20260820)
    idxs = [rng.integers(0, len(dev), len(dev)) for _ in range(200)]

    print(f"  {'design':26s} {'weighted':>8} {'risk-adj':>8}   per tier: score spend/cap P(over)")
    E_ = __import__("exp_sum_quantile")
    E.evaluate = None  # not used; keep the namespace obvious
    from exp_sum_quantile import evaluate
    evaluate("current: sum of quantiles", dev, gain, cost, light_hat, E.route, idxs)
    for z in (2.0, 2.5, 3.0, 4.0):
        evaluate(f"measured sigma, z={z}", dev, gain, cost, light_hat,
                 lambda r, g, c, l, t, z=z: route_sigma(r, g, c, l, t, z, k), idxs)
