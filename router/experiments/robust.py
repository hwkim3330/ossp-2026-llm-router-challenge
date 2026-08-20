# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Budget safety by construction rather than by tuning.

Tuning the margin for maximum CV score produced a dev weighted of 0.4779 -- worse
than routing everything to light -- because the balanced tier landed at 2.28x
against a 2.0x cap and scored zero. The payoff is asymmetric: overrun forfeits the
whole tier, so a few points of score are never worth a real chance of zero. The
margin therefore is not a hyperparameter to maximise over.

Instead: select greedily on a mid cost quantile, then re-price the basket at a
pessimistic quantile and drop the least efficient purchases until even that
pessimistic bill fits. The first quantile decides what looks worth buying; the
second decides how much of it we can afford to be wrong about.
"""
import json, sys, warnings, importlib.util
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
spec=importlib.util.spec_from_file_location("rd",str(Path(__file__).resolve().parent / str(Path(__file__).resolve().parent / "router_dev.py")))
rd=importlib.util.module_from_spec(spec); sys.argv=["x"]; spec.loader.exec_module(rd)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.sparse import hstack

L, UP, POL = rd.LIGHT, rd.UPGRADES, rd.POLICY

def fit(tr, te, quantiles):
    tw=TfidfVectorizer(sublinear_tf=True,ngram_range=(1,2),min_df=2,max_features=30000,strip_accents="unicode")
    tc=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,max_features=40000)
    A=[r["text"] for r in tr]; B=[r["text"] for r in te]
    Xa=hstack([tw.fit_transform(A), tc.fit_transform(A)]).tocsr(); Xb=hstack([tw.transform(B), tc.transform(B)]).tocsr()
    Ha=np.array([rd.hand_features(t) for t in A]); Hb=np.array([rd.hand_features(t) for t in B])
    gain, cost = {}, {q: {} for q in quantiles}
    for m in UP:
        yg=np.array([r["score"][m]-r["score"][L] for r in tr])
        gain[m]=Ridge(alpha=1.0).fit(Xa,yg).predict(Xb)
        yc=np.clip([r["cost"][m]-r["cost"][L] for r in tr],1.0,None)
        base=Ridge(alpha=1.0).fit(Xa,np.log1p(yc)); a,b=base.predict(Xa),base.predict(Xb)
        for q in quantiles:
            mdl=HistGradientBoostingRegressor(loss="quantile",quantile=q,max_iter=250,
                learning_rate=0.06,max_depth=6,random_state=0).fit(np.c_[Ha,a],yc)
            cost[q][m]=mdl.predict(np.c_[Hb,b]).clip(1.0)
    return gain, cost

def route(rows, gain, cost, q_pick, q_safe, tier):
    n=len(rows); base=sum(r["cost"][L] for r in rows)
    cap=float(POL["tiers"][tier]["budget_multiplier"])*base
    picks=[]
    spent=base
    taken=set()
    for eff,i,m in sorted(((gain[m][i]/cost[q_pick][m][i], i, m)
                           for i in range(n) for m in UP if gain[m][i]>0), reverse=True):
        if i in taken: continue
        if spent+cost[q_pick][m][i] <= cap:
            spent += cost[q_pick][m][i]; picks.append((eff,i,m)); taken.add(i)
    # Re-price at the pessimistic quantile and shed the weakest until it fits.
    while picks:
        pess = base + sum(cost[q_safe][m][i] for _,i,m in picks)
        if pess <= cap: break
        picks.pop()
    choice=[L]*n
    for _,i,m in picks: choice[i]=m
    true=sum(rows[i]["cost"][choice[i]] for i in range(n))
    score=sum(rows[i]["score"][choice[i]] for i in range(n))/n
    return (score if true<=cap else 0.0), true/base, true<=cap, choice

if __name__ == "__main__":
    train, dev = rd.load("train"), rd.load("dev")
    QS=(0.5,0.7,0.9,0.95)
    gain, cost = fit(train, dev, QS)
    bs=sum(r["score"][L] for r in dev)/len(dev)
    print(f"dev all-light {bs:.4f}  oracle 0.7974")
    best=None
    for qp in (0.5,0.7):
        for qsafe in (0.9,0.95):
            w=0.0; det=[]
            for tier in POL["tiers"]:
                s,sp,ok,_=route(dev,gain,cost,qp,qsafe,tier)
                w+=float(POL["tiers"][tier]["weight"])*s
                det.append(f"{tier[:4]}={s:.4f}@{sp:.2f}x{'' if ok else '!'}")
            print(f"  pick q={qp} safe q={qsafe}  weighted {w:.4f}  " + " ".join(det))
            if best is None or w>best[0]: best=(w,qp,qsafe)
    print(f"\nbest {best[0]:.4f} at pick={best[1]} safe={best[2]}  captured {(best[0]-bs)/(0.7974-bs):.1%}")
