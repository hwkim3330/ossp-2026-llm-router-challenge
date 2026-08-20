# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""What is the real chance each tier is forfeited, for the shipped design and for
a cost model that only sees length?

`exp_margin.py` measured the spend ratio against the *true* all-light bill, but
the container never sees that bill -- it predicts it, at q_light=0.75, landing
around 0.93x of the truth, which shrinks the cap and makes the router spend less
than those numbers suggest. So the shipped configuration has to be simulated the
way it actually runs, or its risk is overstated.

The comparison is between two cost-model designs, both routed identically:

  A  the shipped one: a log-cost Ridge on the full-text n-grams feeds an `aux`
     column into the quantile model
  B  the quantile model sees hand features and log length, and no n-grams

Reported per tier: dev score, spend ratio, and the share of 200 bootstrap
resamples of dev that exceed the cap -- which is the probability of scoring zero
on that tier, not a margin to admire.
"""
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("rd", str(Path(__file__).resolve().parent / "router_dev.py"))
rd = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(rd)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.sparse import hstack

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
TIERS = list(POL["tiers"])
QS = (0.5, 0.8, 0.9, 0.95)
GAIN_CAP = 500
Q_LIGHT = 0.75
SLACK = 0.03


def vectorise(tr_txt, te_txt, cap):
    tw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                         max_features=30000, strip_accents="unicode")
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                         max_features=40000)
    A = [t[:cap] for t in tr_txt] if cap else tr_txt
    B = [t[:cap] for t in te_txt] if cap else te_txt
    return (hstack([tw.fit_transform(A), tc.fit_transform(A)]).tocsr(),
            hstack([tw.transform(B), tc.transform(B)]).tocsr())


def build(tr, te, cost_uses_text):
    A_full = [r["text"] for r in tr]
    B_full = [r["text"] for r in te]
    La = np.c_[[rd.hand_features(t) for t in A_full], np.log1p([len(t) for t in A_full])]
    Lb = np.c_[[rd.hand_features(t) for t in B_full], np.log1p([len(t) for t in B_full])]
    Xa, Xb = vectorise(A_full, B_full, GAIN_CAP)
    Ca, Cb = vectorise(A_full, B_full, 0) if cost_uses_text else (None, None)

    def quantiles(y, fa, fb):
        return {q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
            max_depth=6, random_state=0).fit(fa, y).predict(fb).clip(1.0)
            for q in QS}

    gain, cost = {}, {q: {} for q in QS}
    for m in UP:
        yg = np.array([r["score"][m] - r["score"][L] for r in tr])
        gain[m] = Ridge(alpha=1.0).fit(Xa, yg).predict(Xb)
        yc = np.clip([r["cost"][m] - r["cost"][L] for r in tr], 1.0, None)
        if cost_uses_text:
            base = Ridge(alpha=1.0).fit(Ca, np.log1p(yc))
            fa, fb = np.c_[La, base.predict(Ca)], np.c_[Lb, base.predict(Cb)]
        else:
            fa, fb = La, Lb
        for q, pred in quantiles(yc, fa, fb).items():
            cost[q][m] = pred

    # the all-light bill the container has to predict for itself
    yl = np.clip([r["cost"][L] for r in tr], 1.0, None)
    if cost_uses_text:
        bl = Ridge(alpha=1.0).fit(Ca, np.log1p(yl))
        fa, fb = np.c_[La, bl.predict(Ca)], np.c_[Lb, bl.predict(Cb)]
    else:
        fa, fb = La, Lb
    light_hat = HistGradientBoostingRegressor(
        loss="quantile", quantile=Q_LIGHT, max_iter=250, learning_rate=0.06,
        max_depth=6, random_state=0).fit(fa, yl).predict(fb).clip(1.0)
    return gain, cost, light_hat


def route(rows, gain, cost, light_hat, q_pick, q_safe, tier):
    """The shipped decision rule: cap from the *estimated* light bill, fill on
    q_pick, re-price on q_safe, hold SLACK back."""
    n = len(rows)
    est = float(light_hat.sum())
    cap = float(POL["tiers"][tier]["budget_multiplier"]) * est
    target = cap * (1.0 - SLACK)
    picks, spent, taken = [], est, set()
    for eff, i, m in sorted(((gain[m][i] / cost[q_pick][m][i], i, m)
                             for i in range(n) for m in UP if gain[m][i] > 0),
                            reverse=True):
        if i in taken:
            continue
        if spent + cost[q_pick][m][i] <= target:
            spent += cost[q_pick][m][i]
            picks.append((eff, i, m))
            taken.add(i)
    while picks and est + sum(cost[q_safe][m][i] for _, i, m in picks) > target:
        picks.pop()
    choice = [L] * n
    for _, i, m in picks:
        choice[i] = m
    true_light = sum(r["cost"][L] for r in rows)
    true_cost = sum(rows[i]["cost"][choice[i]] for i in range(n))
    true_cap = float(POL["tiers"][tier]["budget_multiplier"]) * true_light
    score = sum(rows[i]["score"][choice[i]] for i in range(n)) / n
    return (score if true_cost <= true_cap else 0.0), true_cost / true_light, true_cost <= true_cap


def evaluate(name, dev, gain, cost, light_hat, q_safe, draws=200, seed=20260820):
    rng = np.random.default_rng(seed)
    n = len(dev)
    idxs = [rng.integers(0, n, n) for _ in range(draws)]
    print(f"\n{name}   q_safe={q_safe}")
    weighted = 0.0
    risk_weighted = 0.0
    for tier in TIERS:
        s, sp, ok = route(dev, gain, cost, light_hat, 0.5, q_safe, tier)
        ratios = []
        for idx in idxs:
            sub = [dev[i] for i in idx]
            g = {m: gain[m][idx] for m in UP}
            c = {q: {m: cost[q][m][idx] for m in UP} for q in cost}
            _, r, _ = route(sub, g, c, light_hat[idx], 0.5, q_safe, tier)
            ratios.append(r)
        ratios = np.array(ratios)
        cap = float(POL["tiers"][tier]["budget_multiplier"])
        p_over = float(np.mean(ratios > cap))
        w = float(POL["tiers"][tier]["weight"])
        weighted += w * s
        risk_weighted += w * s * (1 - p_over)
        print(f"  {tier:9s} score {s:.4f}  spend {sp:.2f}/{cap:.2f}  "
              f"bootstrap sd {ratios.std():.3f}  P(over) {p_over:5.1%}"
              + ("" if ok else "   OVER ON DEV"))
    print(f"  weighted {weighted:.4f}   risk-adjusted {risk_weighted:.4f}")
    return weighted, risk_weighted


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    A = build(train, dev, True)
    B = build(train, dev, False)
    results = {}
    results["A@0.95 (shipped)"] = evaluate("A · cost sees full text (shipped)", dev, *A, 0.95)
    for q in (0.8, 0.9, 0.95):
        results[f"B@{q}"] = evaluate("B · cost sees length only", dev, *B, q)
    print("\n" + f"{'config':22s} {'dev weighted':>13} {'risk-adjusted':>14}")
    for k, (w, r) in results.items():
        print(f"{k:22s} {w:13.4f} {r:14.4f}")
