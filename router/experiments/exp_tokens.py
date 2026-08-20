# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Can the cost residual be shrunk? It is 96% output tokens on axk1-think.

Everything left on the table now runs through one number: the spread of the cost
prediction. Premium leaves 1.24x of its cap unspent, and the corrected sqrt(N)
bound showed that is not a pricing artifact -- an `axk1-think` purchase carries a
residual sd of 208,045 credits, so the budget genuinely cannot be committed.

The decomposition says where to aim. Cost is `in_rate*in_tokens +
out_rate*out_tokens`; input tokens are near-deterministic in the prompt
(correlation 0.9985 with character count) and carry 2.8% of the cost-difference
variance for `axk1-think`. The other 96% is output tokens -- how long the model
thinks -- which is the only quantity worth modelling better.

Three things are compared, all scored by out-of-fold residual sd on train:

    1  the current target: the cost difference, on hand features + log length
    2  the same, plus the prompt n-grams
    3  compose: predict each model's log output tokens, take input tokens as a
       function of the text, and rebuild the cost from the published rates

(3) is the interesting one. Regressing the composite hides that two of its three
parts are nearly free, and a model spending capacity on them has less left for
the part that is hard.
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
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix

import exp_gain as E

L, UP = rd.LIGHT, rd.UPGRADES
RATES = {m: (float(v["input_token_rate"]), float(v["output_token_rate"]))
         for m, v in rd.POLICY["models"].items()}


def usage(split):
    blob = json.loads((HERE.parent.parent / "data" / split / "outcomes.json").read_text())
    return {e["episode_id"]: {m: (float(e["models"][m]["input_tokens"]),
                                  float(e["models"][m]["output_tokens"]))
                              for m in RATES} for e in blob["episodes"]}


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                         max_depth=6, random_state=0, **kw)


def run():
    train = rd.load("train")
    use = usage("train")
    texts = [r["text"] for r in train]
    base_feat = np.c_[[rd.hand_features(t) for t in texts],
                      np.log1p([len(t) for t in texts])]
    tok = {m: np.array([use[r["id"]][m] for r in train]) for m in RATES}

    kf = KFold(n_splits=4, shuffle=True, random_state=20260820)
    res = {k: {m: [] for m in UP} for k in ("1 direct", "2 direct+ngrams", "3 composed")}

    for tr_i, te_i in kf.split(base_feat):
        Fa, Fb = base_feat[tr_i], base_feat[te_i]
        Xa, Xb = E.vectorise([texts[i] for i in tr_i], [texts[i] for i in te_i])
        Ga = hstack([csr_matrix(Fa), Xa]).tocsr()
        Gb = hstack([csr_matrix(Fb), Xb]).tocsr()

        for m in UP:
            ir, orr = RATES[m]
            irl, orl = RATES[L]
            true = (ir * tok[m][:, 0] + orr * tok[m][:, 1]
                    - irl * tok[L][:, 0] - orl * tok[L][:, 1])

            p1 = hgb().fit(Fa, true[tr_i]).predict(Fb)
            res["1 direct"][m].append(true[te_i] - p1)

            # n-grams are dense-unfriendly for HGB, so use the linear path the
            # earlier design used: a model on the sparse matrix, added as a feature
            from sklearn.linear_model import Ridge
            aux_a = Ridge(alpha=1.0).fit(Ga, np.log1p(np.clip(true[tr_i], 1, None)))
            p2 = hgb().fit(np.c_[Fa, aux_a.predict(Ga)], true[tr_i]).predict(
                np.c_[Fb, aux_a.predict(Gb)])
            res["2 direct+ngrams"][m].append(true[te_i] - p2)

            # composed: log output tokens per model, input tokens from the text
            parts = {}
            for mm in (L, m):
                lo = hgb().fit(Fa, np.log1p(tok[mm][tr_i, 1])).predict(Fb)
                li = hgb().fit(Fa, np.log1p(tok[mm][tr_i, 0])).predict(Fb)
                parts[mm] = (np.expm1(li), np.expm1(lo))
            p3 = (ir * parts[m][0] + orr * parts[m][1]
                  - irl * parts[L][0] - orl * parts[L][1])
            res["3 composed"][m].append(true[te_i] - p3)

    print(f"  {'model':20s} " + "  ".join(f"{m:>18}" for m in UP))
    print("  out-of-fold residual sd of the cost difference, in credits\n")
    for k, per in res.items():
        cells = []
        for m in UP:
            r = np.concatenate(per[m])
            cells.append(f"{r.std():18,.0f}")
        print(f"  {k:20s} " + "  ".join(cells))

    print("\n  out-of-fold median absolute error")
    for k, per in res.items():
        cells = [f"{np.median(np.abs(np.concatenate(per[m]))):18,.0f}" for m in UP]
        print(f"  {k:20s} " + "  ".join(cells))


if __name__ == "__main__":
    run()
