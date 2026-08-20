# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Budget-constrained router: predict gain and cost per model, then fill each tier.

The task is not "which model is best" -- it is a multiple-choice knapsack. Every
episode may be routed to ax31 or axk1-think instead of ax31-light, each purchase
costs credits, and each tier caps total spend at a multiple of the all-light
bill. Exceeding a cap scores zero for that tier, so cost has to be predicted, not
just quality: the bill is computed from the chosen model's *actual* token usage,
which is unknown at routing time.

So two predictors per upgrade model:
  * expected score gain over ax31-light
  * expected extra cost, driven almost entirely by output tokens

and then a greedy fill by predicted gain per predicted credit, which is the
standard LP-relaxation solution to this knapsack and is near-optimal at n=1760.

Measured ceilings on train (perfect foresight, same greedy): fast 0.7339,
balanced 0.7886, premium 0.8452, weighted 0.7837 against an all-light 0.5973.
Those are what a router is scored against, not 1.0 -- axk1-think costs 23x light,
so even the premium tier can only upgrade a third of the set.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent.parent
POLICY = json.loads((ROOT / "configs/routing-policy.v1.json").read_text())
RATES = {m: (Decimal(v["input_token_rate"]), Decimal(v["output_token_rate"]))
         for m, v in POLICY["models"].items()}
LIGHT = POLICY["light_model_id"]
UPGRADES = [m for m in RATES if m != LIGHT]

CODE = re.compile(r"```|\bdef \b|\bclass \b|\bimport \b|[{};]|</|/>")
MATH = re.compile(r"[=+\-*/^]|\\frac|\\sum|\\int|\$|\b\d+\s*[+\-*/]\s*\d+")
HANGUL = re.compile(r"[가-힣]")
CJK = re.compile(r"[一-鿿぀-ヿ]")


def episode_text(ep: dict) -> str:
    if "prompt" in ep and ep["prompt"]:
        return str(ep["prompt"])
    parts = []
    for msg in ep.get("messages", []):
        parts.append(f"{msg.get('role','')}: {msg.get('content','')}")
    return "\n".join(parts)


def hand_features(text: str) -> list[float]:
    n = max(len(text), 1)
    words = text.split()
    digits = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)
    return [
        len(text), len(words), np.log1p(len(text)),
        digits / n, upper / n,
        len(HANGUL.findall(text)) / n,
        len(CJK.findall(text)) / n,
        float(bool(CODE.search(text))), len(CODE.findall(text)) / n * 100,
        float(bool(MATH.search(text))), len(MATH.findall(text)) / n * 100,
        text.count("?"), text.count("\n"),
        float("step by step" in text.lower()), float("prove" in text.lower()),
        float("explain" in text.lower()), float("translate" in text.lower()),
        np.mean([len(w) for w in words]) if words else 0.0,
    ]


def cost_of(model: str, usage: dict) -> float:
    ir, orr = RATES[model]
    return float(ir * usage["input_tokens"] + orr * usage["output_tokens"])


def load(split: str):
    d = ROOT / "data" / split
    inputs = {e["episode_id"]: e for e in json.loads((d / "inputs-base.json").read_text())["episodes"]}
    extra = d / "aime-selection.json"
    if extra.is_file():
        blob = json.loads(extra.read_text())
        for e in blob.get("episodes", []):
            inputs.setdefault(e["episode_id"], e)
    outcomes = json.loads((d / "outcomes.json").read_text())["episodes"]
    rows = []
    for oc in outcomes:
        ep = inputs.get(oc["episode_id"])
        if ep is None:
            continue
        rows.append({
            "id": oc["episode_id"],
            "text": episode_text(ep),
            "score": {m: float(oc["models"][m]["score"]) for m in RATES},
            "cost": {m: cost_of(m, oc["models"][m]) for m in RATES},
        })
    return rows


def greedy_fill(gain: dict, cost: dict, true_score, true_cost, budget: float, n: int):
    """Buy upgrades by predicted efficiency; bill at the true cost of what was bought."""
    spent = sum(true_cost[i][LIGHT] for i in range(n))
    total = sum(true_score[i][LIGHT] for i in range(n))
    choice = [LIGHT] * n
    cands = []
    for i in range(n):
        for m in UPGRADES:
            g, c = gain[m][i], cost[m][i]
            if g > 0 and c > 0:
                cands.append((g / c, i, m))
    cands.sort(reverse=True)
    taken = set()
    for _, i, m in cands:
        if i in taken:
            continue
        dc = true_cost[i][m] - true_cost[i][LIGHT]
        if spent + dc <= budget:
            spent += dc
            total += true_score[i][m] - true_score[i][LIGHT]
            choice[i] = m
            taken.add(i)
    return total / n, spent, choice


def main() -> None:
    train, dev = load("train"), load("dev")
    print(f"train {len(train)}  dev {len(dev)}")

    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                            max_features=20000, strip_accents="unicode")
    Xtr_t = tfidf.fit_transform([r["text"] for r in train])
    Xdv_t = tfidf.transform([r["text"] for r in dev])
    Htr = np.array([hand_features(r["text"]) for r in train])
    Hdv = np.array([hand_features(r["text"]) for r in dev])

    pred_gain, pred_cost = {}, {}
    for m in UPGRADES:
        y_gain = np.array([r["score"][m] - r["score"][LIGHT] for r in train])
        lin = Ridge(alpha=1.0).fit(Xtr_t, y_gain)
        g_lin_tr, g_lin_dv = lin.predict(Xtr_t), lin.predict(Xdv_t)
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                           max_depth=6, random_state=0)
        gb.fit(np.c_[Htr, g_lin_tr], y_gain)
        pred_gain[m] = gb.predict(np.c_[Hdv, g_lin_dv])

        # Cost is dominated by output tokens and spans orders of magnitude, so
        # predict it in log space. Some upgrades are *cheaper* than light -- the
        # bigger model answers concisely where light rambles -- so the delta can
        # be negative and log1p would return NaN. Floor it at 1 credit: those
        # episodes are effectively free upgrades and the greedy fill should rank
        # them at the very top, which a floor of 1 achieves.
        y_cost = np.log1p(np.clip([r["cost"][m] - r["cost"][LIGHT] for r in train], 1.0, None))
        cb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                           max_depth=6, random_state=0)
        cb.fit(np.c_[Htr, g_lin_tr], y_cost)
        pred_cost[m] = np.expm1(cb.predict(np.c_[Hdv, g_lin_dv])).clip(1.0)

    n = len(dev)
    true_score = [r["score"] for r in dev]
    true_cost = [r["cost"] for r in dev]
    base_cost = sum(c[LIGHT] for c in true_cost)
    base_score = sum(s[LIGHT] for s in true_score) / n

    oracle_gain = {m: np.array([r["score"][m] - r["score"][LIGHT] for r in dev]) for m in UPGRADES}
    oracle_cost = {m: np.array([max(r["cost"][m] - r["cost"][LIGHT], 1.0) for r in dev]) for m in UPGRADES}

    print(f"\ndev all-light = {base_score:.4f}")
    print(f"{'tier':<10}{'budget':>8}{'router':>10}{'oracle':>10}{'captured':>10}{'spend':>9}")
    weighted = weighted_oracle = 0.0
    for tier, cfg in POLICY["tiers"].items():
        b = float(cfg["budget_multiplier"]) * base_cost
        w = float(cfg["weight"])
        s_r, spend_r, _ = greedy_fill(pred_gain, pred_cost, true_score, true_cost, b, n)
        s_o, _, _ = greedy_fill(oracle_gain, oracle_cost, true_score, true_cost, b, n)
        frac = (s_r - base_score) / (s_o - base_score) if s_o > base_score else 0.0
        weighted += w * s_r
        weighted_oracle += w * s_o
        print(f"{tier:<10}{cfg['budget_multiplier']:>8}{s_r:>10.4f}{s_o:>10.4f}"
              f"{frac:>9.1%}{spend_r/base_cost:>8.2f}x")
    print(f"\nweighted: router {weighted:.4f}   oracle {weighted_oracle:.4f}   "
          f"all-light {base_score:.4f}")
    print(f"captured {(weighted - base_score) / (weighted_oracle - base_score):.1%} of the headroom")


if __name__ == "__main__":
    main()
