# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Does truncating the vectoriser input cost score?

The 90 s limit is a binary risk -- a tier that times out scores zero -- and 72%
of the runtime is the TF-IDF transform. The length distribution says that is
almost entirely a tail problem: the median episode is 237 characters, but the
top 9% hold 90% of all characters (p99 = 65,327, max = 71,094). Capping at 2,000
leaves 9.8% of the characters to vectorise and touches 9.1% of documents.

Two things are deliberately kept off the cap:

* `hand_features` still sees the full text, so length -- which is the strongest
  cost signal there is -- is not thrown away. Only the n-gram input shrinks.
* the fit and the transform use the same cap, so train and dev stay comparable.

Baseline to beat: dev weighted 0.6748 at pick q=0.5, safe q=0.95.
"""
import importlib.util, sys, time, warnings
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

import robust  # reuse the routing and scoring, unchanged

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY
QS = (0.5, 0.95)


def fit(tr, te, cap, char_range=(3, 5)):
    tw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                         max_features=30000, strip_accents="unicode")
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=char_range, min_df=3,
                         max_features=40000)
    A_full = [r["text"] for r in tr]
    B_full = [r["text"] for r in te]
    A = [t[:cap] for t in A_full] if cap else A_full
    B = [t[:cap] for t in B_full] if cap else B_full

    t0 = time.time()
    Xa = hstack([tw.fit_transform(A), tc.fit_transform(A)]).tocsr()
    t_fit = time.time() - t0
    t0 = time.time()
    Xb = hstack([tw.transform(B), tc.transform(B)]).tocsr()
    t_tr = time.time() - t0

    # hand features on the FULL text: length is the cost signal, keep it
    Ha = np.array([rd.hand_features(t) for t in A_full])
    Hb = np.array([rd.hand_features(t) for t in B_full])

    gain, cost = {}, {q: {} for q in QS}
    for m in UP:
        yg = np.array([r["score"][m] - r["score"][L] for r in tr])
        gain[m] = Ridge(alpha=1.0).fit(Xa, yg).predict(Xb)
        yc = np.clip([r["cost"][m] - r["cost"][L] for r in tr], 1.0, None)
        base = Ridge(alpha=1.0).fit(Xa, np.log1p(yc))
        a, b = base.predict(Xa), base.predict(Xb)
        for q in QS:
            mdl = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
                max_depth=6, random_state=0).fit(np.c_[Ha, a], yc)
            cost[q][m] = mdl.predict(np.c_[Hb, b]).clip(1.0)
    return gain, cost, t_fit, t_tr, Xb.shape


def weighted(dev, gain, cost):
    total, detail = 0.0, []
    for tier in POL["tiers"]:
        s, sp, ok, _ = robust.route(dev, gain, cost, 0.5, 0.95, tier)
        total += float(POL["tiers"][tier]["weight"]) * s
        detail.append(f"{tier[:4]}={s:.4f}@{sp:.2f}x{'' if ok else '!'}")
    return total, " ".join(detail)


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    print(f"{'cap':>7} {'weighted':>9} {'fit s':>7} {'xform s':>8}  tiers")
    for cap in (0, 8000, 4000, 2000, 1000, 500, 250):
        gain, cost, t_fit, t_tr, shape = fit(train, dev, cap)
        w, detail = weighted(dev, gain, cost)
        label = "none" if cap == 0 else str(cap)
        print(f"{label:>7} {w:9.4f} {t_fit:7.2f} {t_tr:8.2f}  {detail}")
