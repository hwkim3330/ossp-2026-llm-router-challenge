# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Does the cost model need the text at all, or only its length?

`exp_cap.py` showed truncating the vectoriser input pushes the balanced tier over
its 2.0x cap at nearly every cap tried, so the loss is in the cost path, not the
gain path. That points at one link: the log-cost Ridge is fitted on the sparse
text matrix, and its prediction is the `aux` column the quantile model leans on.
Truncation removes the length information that column was carrying.

If cost is mostly a function of length, the cost path can be given length
directly and stop depending on the n-grams -- which would let the gain path be
truncated freely.

Reported for each variant: dev weighted, and the spend/cap ratio per tier, since
a tier at 1.99x of a 2.0x cap has passed by luck rather than by design.
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
QS = (0.5, 0.95)


def vectorise(tr_txt, te_txt, cap):
    tw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                         max_features=30000, strip_accents="unicode")
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                         max_features=40000)
    A = [t[:cap] for t in tr_txt] if cap else tr_txt
    B = [t[:cap] for t in te_txt] if cap else te_txt
    return (hstack([tw.fit_transform(A), tc.fit_transform(A)]).tocsr(),
            hstack([tw.transform(B), tc.transform(B)]).tocsr())


def build(tr, te, gain_cap, cost_uses_text):
    A_full = [r["text"] for r in tr]
    B_full = [r["text"] for r in te]
    Ha = np.array([rd.hand_features(t) for t in A_full])
    Hb = np.array([rd.hand_features(t) for t in B_full])
    # length is the one feature truncation would destroy, so state it explicitly
    La = np.c_[Ha, np.log1p([len(t) for t in A_full])]
    Lb = np.c_[Hb, np.log1p([len(t) for t in B_full])]

    Xa, Xb = vectorise(A_full, B_full, gain_cap)
    if cost_uses_text:
        Ca, Cb = vectorise(A_full, B_full, 0)   # full text for the cost path only
    else:
        Ca = Cb = None

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


def report(label, dev, gain, cost, slack=0.0):
    total, detail = 0.0, []
    for tier in POL["tiers"]:
        s, sp, ok, _ = robust.route(dev, gain, cost, 0.5, 0.95, tier)
        total += float(POL["tiers"][tier]["weight"]) * s
        cap = float(POL["tiers"][tier]["budget_multiplier"])
        detail.append(f"{tier[:4]} {s:.4f} {sp:.2f}/{cap:.2f} ({(cap - sp) / cap:+.0%})"
                      + ("" if ok else " OVER"))
    print(f"{label:38s} {total:.4f}  " + "  ".join(detail))
    return total


if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    print(f"{'variant':38s} {'weighted':8s}  per-tier score spend/cap (margin)")
    report("baseline: full text both paths", dev, *build(train, dev, 0, True))
    report("cost on length only, gain full", dev, *build(train, dev, 0, False))
    for cap in (2000, 500):
        report(f"cost on length only, gain cap {cap}", dev,
               *build(train, dev, cap, False))
        report(f"cost full text,      gain cap {cap}", dev,
               *build(train, dev, cap, True))
