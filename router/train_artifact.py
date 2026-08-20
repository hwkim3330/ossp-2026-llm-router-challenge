# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Train the router on all public data and serialise it for the container.

The container gets prompts only -- no outcomes, no scores -- and cannot train, so
everything it needs is fitted here and baked into the image: the two TF-IDF
vocabularies, a Ridge gain model per upgrade, and a quantile cost model per
upgrade at the fill and safety quantiles.

Fitted on train+dev together (2,640 episodes) because the held-out numbers have
already been read off dev and the final artifact should use every labelled
episode available. The dev-only configuration is what the reported 0.6748 was
measured with; see FINDINGS.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
from decimal import Decimal
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

from router_features import episode_text, hand_features

Q_FILL, Q_SAFE = 0.5, 0.95


def load(challenge_root: Path, split: str, policy: dict):
    rates = {m: (Decimal(v["input_token_rate"]), Decimal(v["output_token_rate"]))
             for m, v in policy["models"].items()}
    d = challenge_root / "data" / split
    inputs = {e["episode_id"]: e
              for e in json.loads((d / "inputs-base.json").read_text())["episodes"]}
    extra = d / "aime-selection.json"
    if extra.is_file():
        for e in json.loads(extra.read_text()).get("episodes", []):
            inputs.setdefault(e["episode_id"], e)
    rows = []
    for oc in json.loads((d / "outcomes.json").read_text())["episodes"]:
        ep = inputs.get(oc["episode_id"])
        if ep is None:
            continue
        cost = {}
        for m, (ir, orr) in rates.items():
            u = oc["models"][m]
            cost[m] = float(ir * u["input_tokens"] + orr * u["output_tokens"])
        rows.append({"text": episode_text(ep),
                     "score": {m: float(oc["models"][m]["score"]) for m in rates},
                     "cost": cost})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge-root", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="repository root holding data/ (default: this checkout)")
    ap.add_argument("--out", type=Path, default=Path("artifact/router.pkl"))
    args = ap.parse_args()

    policy = json.loads((args.challenge_root / "configs/routing-policy.v1.json").read_text())
    light = policy["light_model_id"]
    upgrades = [m for m in policy["models"] if m != light]

    rows = load(args.challenge_root, "train", policy) + load(args.challenge_root, "dev", policy)
    print(f"fitting on {len(rows)} episodes")

    texts = [r["text"] for r in rows]
    tw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                         max_features=30000, strip_accents="unicode")
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=40000)
    X = hstack([tw.fit_transform(texts), tc.fit_transform(texts)]).tocsr()
    H = np.array([hand_features(t) for t in texts])

    gain, cost_models = {}, {q: {} for q in (Q_FILL, Q_SAFE)}
    for m in upgrades:
        y = np.array([r["score"][m] - r["score"][light] for r in rows])
        gain[m] = Ridge(alpha=1.0).fit(X, y)
        yc = np.clip([r["cost"][m] - r["cost"][light] for r in rows], 1.0, None)
        base = Ridge(alpha=1.0).fit(X, np.log1p(yc))
        aux = base.predict(X)
        for q in (Q_FILL, Q_SAFE):
            cost_models[q][m] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
                max_depth=6, random_state=0).fit(np.c_[H, aux], yc)
        cost_models.setdefault("base", {})[m] = base
        print(f"  {m}: gain + cost models fitted")

    # The cap is a multiple of the all-light bill, and the container is never told
    # what that bill is -- it sees prompts only. Predicting it too high inflates the
    # cap and risks an overrun, which forfeits the tier, while predicting it too low
    # only costs some score. So light's own cost is fitted at a LOW quantile and the
    # asymmetry is resolved in the safe direction.
    y_light = np.array([r["cost"][light] for r in rows])
    base_light = Ridge(alpha=1.0).fit(X, np.log1p(y_light))
    light_models = {"base": base_light}
    for q in (0.25, 0.5, 0.75):
        light_models[q] = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=250, learning_rate=0.06,
            max_depth=6, random_state=0).fit(np.c_[H, base_light.predict(X)], y_light)
    print(f"  {light}: cost model fitted (mean {y_light.mean():.0f} credits/episode)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fh:
        pickle.dump({"tfidf_word": tw, "tfidf_char": tc, "gain": gain,
                     "cost": cost_models, "light": light, "upgrades": upgrades,
                     "light_cost": light_models, "q_light": 0.75,
                     "policy": policy, "q_fill": Q_FILL, "q_safe": Q_SAFE}, fh,
                    protocol=4)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
