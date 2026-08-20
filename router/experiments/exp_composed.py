# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Does a physically composed cost prediction buy anything end to end?

`exp_tokens.py` found the split that matters. Building the cost from its parts --
log output tokens per model, input tokens from the text, recombined with the
published rates -- cuts the out-of-fold median absolute error by 34% on
`axk1-think` (30,973 -> 20,315 credits) and 26% on `ax31`, while the residual sd
barely moves (194,339 -> 191,773). The bulk gets much better and the tail does
not, which is what a kurtosis near 100 implies.

That predicts a specific, asymmetric outcome, worth stating before measuring: the
greedy fills on the median, so a better median should choose better purchases and
score more, while the safety pass is driven by the spread, so the spend ceiling
should not move. If instead the spend ceiling rises, something is wrong.

Note the composed predictor adds no new information -- it reads the same hand
features and length. What it adds is the cost identity as a structural prior,
which a single regression on the composite has to discover for itself.
"""
import importlib.util, json, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rd", str(HERE / "router_dev.py"))
rd = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(rd)

from sklearn.ensemble import HistGradientBoostingRegressor

import exp_gain as E
from exp_sum_quantile import evaluate

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
RATES = {m: (float(v["input_token_rate"]), float(v["output_token_rate"]))
         for m, v in POL["models"].items()}
QS = (0.5, 0.9)


def usage(split):
    blob = json.loads((HERE.parent.parent / "data" / split / "outcomes.json").read_text())
    return {e["episode_id"]: {m: (float(e["models"][m]["input_tokens"]),
                                  float(e["models"][m]["output_tokens"]))
                              for m in RATES} for e in blob["episodes"]}


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                         max_depth=6, random_state=0, **kw)


def composed_cost_models(tr, te):
    """Design B's quantile models, given the composed point estimate as a feature."""
    use_tr = usage("train")
    Fa, Fb = E.cost_features(tr), E.cost_features(te)
    tok = {m: np.array([use_tr[r["id"]][m] for r in tr]) for m in RATES}

    point = {}
    parts = {}
    for m in RATES:
        li = hgb().fit(Fa, np.log1p(tok[m][:, 0]))
        lo = hgb().fit(Fa, np.log1p(tok[m][:, 1]))
        parts[m] = ((np.expm1(li.predict(Fa)), np.expm1(lo.predict(Fa))),
                    (np.expm1(li.predict(Fb)), np.expm1(lo.predict(Fb))))
    irl, orl = RATES[L]
    for m in UP:
        ir, orr = RATES[m]
        a = (ir * parts[m][0][0] + orr * parts[m][0][1]
             - irl * parts[L][0][0] - orl * parts[L][0][1])
        b = (ir * parts[m][1][0] + orr * parts[m][1][1]
             - irl * parts[L][1][0] - orl * parts[L][1][1])
        point[m] = (a, b)

    cost = {q: {} for q in QS}
    for m in UP:
        yc = np.clip([r["cost"][m] - r["cost"][L] for r in tr], 1.0, None)
        a, b = point[m]
        for q in QS:
            cost[q][m] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
                max_depth=6, random_state=0).fit(np.c_[Fa, a], yc).predict(
                np.c_[Fb, b]).clip(1.0)

    yl = np.clip([r["cost"][L] for r in tr], 1.0, None)
    la = irl * parts[L][0][0] + orl * parts[L][0][1]
    lb = irl * parts[L][1][0] + orl * parts[L][1][1]
    light = HistGradientBoostingRegressor(
        loss="quantile", quantile=E.Q_LIGHT, max_iter=250, learning_rate=0.06,
        max_depth=6, random_state=0).fit(np.c_[Fa, la], yl).predict(
        np.c_[Fb, lb]).clip(1.0)
    return cost, light


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    Xa, Xb = E.vectorise([r["text"] for r in train], [r["text"] for r in dev])
    gain = E.gain_A(Xa, Xb, train)
    rng = np.random.default_rng(20260820)
    idxs = [rng.integers(0, len(dev), len(dev)) for _ in range(200)]

    print(f"  {'cost model':28s} {'weighted':>8} {'risk-adj':>8}   per tier: score spend/cap P(over)")
    cost, light = E.cost_models(train, dev)
    evaluate("B: hand features + length", dev, gain, cost, light, E.route, idxs)
    cost2, light2 = composed_cost_models(train, dev)
    evaluate("B + composed point estimate", dev, gain, cost2, light2, E.route, idxs)
