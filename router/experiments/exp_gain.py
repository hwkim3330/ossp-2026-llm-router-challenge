# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Is "where does light fail" a better gain target than the pairwise difference?

The gain target is mostly zeros -- exactly 0 for 75% of episodes with `ax31` and
61% with `axk1-think` -- and what is left sits almost entirely on the episodes
the light model gets wrong:

    E[gain | light fails]  ax31 +0.271   axk1-think +0.617
    E[gain | light ok]     ax31 -0.022   axk1-think -0.005

So a regression fitted on the raw difference spends most of its capacity on rows
that carry no signal, while the quantity that actually decides a purchase --
"will light fail here, and would this model fix it" -- is a binary problem with a
35% positive rate.

Candidates, all routed identically (cost on length, q_pick 0.5, q_safe 0.90) so
only the gain estimator changes:

    A  Ridge on the difference                       (current)
    B  P(light fails) x the constant conditional mean
    C  P(light fails) x a Ridge fitted on the failing subset only
    D  P(model correct) - P(light correct), both calibrated

Selection is not made on the dev score alone. Each candidate is also scored by
4-fold CV *within train* on the efficiency ranking the greedy consumes, and only
a candidate that wins both is worth adopting -- picking on dev alone is how the
previous two regressions got chosen.
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

from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
TIERS = list(POL["tiers"])
QS = (0.5, 0.9)
GAIN_CAP, Q_LIGHT, SLACK, Q_FILL, Q_SAFE = 500, 0.75, 0.03, 0.5, 0.9


def vectorise(a_txt, b_txt):
    tw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                         max_features=30000, strip_accents="unicode")
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                         max_features=40000)
    A = [t[:GAIN_CAP] for t in a_txt]
    B = [t[:GAIN_CAP] for t in b_txt]
    return (hstack([tw.fit_transform(A), tc.fit_transform(A)]).tocsr(),
            hstack([tw.transform(B), tc.transform(B)]).tocsr())


def cost_features(rows):
    txt = [r["text"] for r in rows]
    return np.c_[[rd.hand_features(t) for t in txt], np.log1p([len(t) for t in txt])]


# ----------------------------------------------------------------- gain models
def gain_A(Xa, Xb, tr):
    return {m: Ridge(alpha=1.0).fit(
        Xa, [r["score"][m] - r["score"][L] for r in tr]).predict(Xb) for m in UP}


def _p_fail(Xa, Xb, tr):
    y = (np.array([r["score"][L] for r in tr]) < 0.5).astype(int)
    return LogisticRegression(C=1.0, max_iter=2000).fit(Xa, y).predict_proba(Xb)[:, 1]


def gain_B(Xa, Xb, tr):
    pf = _p_fail(Xa, Xb, tr)
    fails = np.array([r["score"][L] for r in tr]) < 0.5
    out = {}
    for m in UP:
        d = np.array([r["score"][m] - r["score"][L] for r in tr])
        out[m] = pf * d[fails].mean()
    return out


def gain_C(Xa, Xb, tr):
    pf = _p_fail(Xa, Xb, tr)
    fails = np.array([r["score"][L] for r in tr]) < 0.5
    idx = np.flatnonzero(fails)
    out = {}
    for m in UP:
        d = np.array([r["score"][m] - r["score"][L] for r in tr])
        # fitted on the failing rows only: the question is "would this model fix
        # it", which is undefined on rows light already gets right
        cond = Ridge(alpha=1.0).fit(Xa[idx], d[idx]).predict(Xb)
        out[m] = pf * np.clip(cond, 0.0, 1.0)
    return out


def gain_D(Xa, Xb, tr):
    def p_ok(model):
        y = (np.array([r["score"][model] for r in tr]) >= 0.5).astype(int)
        return LogisticRegression(C=1.0, max_iter=2000).fit(Xa, y).predict_proba(Xb)[:, 1]
    pl = p_ok(L)
    return {m: p_ok(m) - pl for m in UP}


MODELS = {"A Ridge on difference": gain_A,
          "B P(fail) x constant": gain_B,
          "C P(fail) x conditional Ridge": gain_C,
          "D calibrated P(ok) difference": gain_D}


# -------------------------------------------------------------------- routing
def cost_models(tr, te):
    Fa, Fb = cost_features(tr), cost_features(te)
    cost = {q: {} for q in QS}
    for m in UP:
        yc = np.clip([r["cost"][m] - r["cost"][L] for r in tr], 1.0, None)
        for q in QS:
            cost[q][m] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
                max_depth=6, random_state=0).fit(Fa, yc).predict(Fb).clip(1.0)
    yl = np.clip([r["cost"][L] for r in tr], 1.0, None)
    light = HistGradientBoostingRegressor(
        loss="quantile", quantile=Q_LIGHT, max_iter=250, learning_rate=0.06,
        max_depth=6, random_state=0).fit(Fa, yl).predict(Fb).clip(1.0)
    return cost, light


def route(rows, gain, cost, light_hat, tier):
    n = len(rows)
    est = float(light_hat.sum())
    target = float(POL["tiers"][tier]["budget_multiplier"]) * est * (1.0 - SLACK)
    picks, spent, taken = [], est, set()
    for eff, i, m in sorted(((gain[m][i] / cost[Q_FILL][m][i], i, m)
                             for i in range(n) for m in UP if gain[m][i] > 0),
                            reverse=True):
        if i in taken:
            continue
        if spent + cost[Q_FILL][m][i] <= target:
            spent += cost[Q_FILL][m][i]
            picks.append((eff, i, m))
            taken.add(i)
    while picks and est + sum(cost[Q_SAFE][m][i] for _, i, m in picks) > target:
        picks.pop()
    choice = [L] * n
    for _, i, m in picks:
        choice[i] = m
    tl = sum(r["cost"][L] for r in rows)
    tc = sum(rows[i]["cost"][choice[i]] for i in range(n))
    cap = float(POL["tiers"][tier]["budget_multiplier"]) * tl
    s = sum(rows[i]["score"][choice[i]] for i in range(n)) / n
    return (s if tc <= cap else 0.0), tc / tl, tc <= cap


def cv_ranking(train, folds=4):
    """Spearman of predicted vs true efficiency, inside train. No dev involved."""
    out = {name: {m: [] for m in UP} for name in MODELS}
    kf = KFold(n_splits=folds, shuffle=True, random_state=20260820)
    rows = np.array(train, dtype=object)
    for tr_i, te_i in kf.split(rows):
        tr, te = list(rows[tr_i]), list(rows[te_i])
        Xa, Xb = vectorise([r["text"] for r in tr], [r["text"] for r in te])
        true_cost = {m: np.clip([r["cost"][m] - r["cost"][L] for r in te], 1.0, None)
                     for m in UP}
        true_gain = {m: np.array([r["score"][m] - r["score"][L] for r in te]) for m in UP}
        for name, fn in MODELS.items():
            pred = fn(Xa, Xb, tr)
            for m in UP:
                rho = spearmanr(pred[m] / true_cost[m],
                                true_gain[m] / true_cost[m]).statistic
                out[name][m].append(rho)
    return {n: {m: float(np.mean(v)) for m, v in d.items()} for n, d in out.items()}


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    print("4-fold CV inside train: Spearman of the efficiency ranking\n")
    cv = cv_ranking(train)
    for name, per in cv.items():
        print(f"  {name:32s} " + "  ".join(f"{m} {v:+.3f}" for m, v in per.items()))

    Xa, Xb = vectorise([r["text"] for r in train], [r["text"] for r in dev])
    cost, light_hat = cost_models(train, dev)
    rng = np.random.default_rng(20260820)
    idxs = [rng.integers(0, len(dev), len(dev)) for _ in range(200)]

    print("\nheld-out dev, identical cost model and routing\n")
    print(f"  {'gain model':32s} {'weighted':>9} {'risk-adj':>9}   per-tier spend  P(over)")
    for name, fn in MODELS.items():
        gain = fn(Xa, Xb, train)
        w = radj = 0.0
        cells = []
        for tier in TIERS:
            s, sp, ok = route(dev, gain, cost, light_hat, tier)
            ratios = np.array([route([dev[i] for i in ix],
                                     {m: gain[m][ix] for m in UP},
                                     {q: {m: cost[q][m][ix] for m in UP} for q in cost},
                                     light_hat[ix], tier)[1] for ix in idxs])
            cap = float(POL["tiers"][tier]["budget_multiplier"])
            p_over = float(np.mean(ratios > cap))
            wt = float(POL["tiers"][tier]["weight"])
            w += wt * s
            radj += wt * s * (1 - p_over)
            cells.append(f"{tier[:4]} {sp:.2f}/{cap:.2f} {p_over:.0%}")
        print(f"  {name:32s} {w:9.4f} {radj:9.4f}   " + "  ".join(cells))
