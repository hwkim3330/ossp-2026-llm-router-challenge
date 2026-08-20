# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""How much margin does the balanced tier actually need, and what buys it?

Two results from `exp_cost_path.py` set this up:

* the gain path can be truncated to 500 characters for free (0.6755 vs 0.6748),
  so the n-gram cost is only paid because the *cost* path wants the full text;
* dropping the text from the cost path costs 0.006 of score and takes the
  balanced margin from +4% to +22%.

+4% is the number to worry about. Dev is 880 episodes and the graded split is a
different sample, so the question is not "did balanced pass here" but "how far
does the spend ratio move between samples of this size". That is measurable:
bootstrap dev and look at the spread of spend/cap.

The earlier lesson stands -- tuning a margin for expected score scored 0.4779,
below routing everything to light -- so nothing here is selected for score. The
quantile sweep is reported so the shape is visible, and the choice is made on
whether a design's margin exceeds the measured sampling spread.
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
import robust

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
QS = (0.5, 0.7, 0.8, 0.9, 0.95)
GAIN_CAP = 500


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
        for q in QS:
            mdl = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
                max_depth=6, random_state=0).fit(fa, yc)
            cost[q][m] = mdl.predict(fb).clip(1.0)
    return gain, cost


def sweep(name, dev, gain, cost):
    print(f"\n{name}")
    print(f"  {'q_safe':>6} {'weighted':>9}   " +
          "  ".join(f"{t[:4]:>22}" for t in POL["tiers"]))
    rows = {}
    for q_safe in QS[1:]:
        total, cells = 0.0, []
        for tier in POL["tiers"]:
            s, sp, ok, _ = robust.route(dev, gain, cost, 0.5, q_safe, tier)
            total += float(POL["tiers"][tier]["weight"]) * s
            cap = float(POL["tiers"][tier]["budget_multiplier"])
            cells.append(f"{s:.4f} {sp:.2f}/{cap:.2f} {(cap - sp) / cap:+4.0%}"
                         + ("!" if not ok else " "))
        rows[q_safe] = total
        print(f"  {q_safe:6} {total:9.4f}   " + "  ".join(f"{c:>22}" for c in cells))
    return rows


def bootstrap_spread(dev, gain, cost, q_safe, tier, draws=200, seed=20260820):
    """Spread of the spend ratio over resamples of dev at the same size.

    Routing is a global knapsack, so a subsample has to be routed on its own --
    the ratio is not an average of per-episode quantities.
    """
    rng = np.random.default_rng(seed)
    n = len(dev)
    ratios = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        sub = [dev[i] for i in idx]
        g = {m: gain[m][idx] for m in UP}
        c = {q: {m: cost[q][m][idx] for m in UP} for q in cost}
        _, sp, _, _ = robust.route(sub, g, c, 0.5, q_safe, tier)
        ratios.append(sp)
    return np.array(ratios)


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    designs = {
        "A · cost sees full text (shipped design)": build(train, dev, True),
        "B · cost sees length and hand features only": build(train, dev, False),
    }
    for name, (gain, cost) in designs.items():
        sweep(name, dev, gain, cost)

    print("\nsampling spread of the balanced spend ratio (200 bootstraps of dev, n=880)")
    for name, (gain, cost) in designs.items():
        for q_safe in (0.9, 0.95):
            r = bootstrap_spread(dev, gain, cost, q_safe, "balanced")
            over = float(np.mean(r > 2.0))
            print(f"  {name[:3]} q_safe={q_safe}  mean {r.mean():.3f}  "
                  f"sd {r.std():.3f}  p95 {np.percentile(r, 95):.3f}  "
                  f"max {r.max():.3f}  P(over 2.0) {over:.1%}")
