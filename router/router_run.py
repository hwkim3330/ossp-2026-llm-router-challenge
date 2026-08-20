# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""router-run: pick one model per episode for a single tier, under its budget.

Called once per tier by the evaluator:

    router-run --input /challenge/input/inputs.json \
               --tier fast --output /challenge/output/submission.json

Prompts are the only input; there are no scores or token counts at runtime, so
every model is fitted beforehand and loaded from the baked artifact.

The budget is the whole problem. A tier over its cap scores zero, and the bill is
computed from the chosen model's actual tokens, which are unknown here. So cost is
predicted at two quantiles: purchases are filled greedily on the 0.5 quantile by
predicted gain per predicted credit, then the basket is re-priced at the 0.90
quantile and the weakest purchases dropped until even that pessimistic bill fits,
with 3% of the cap held back as slack. Tuning a margin for score instead scored
below routing everything to light -- see FINDINGS.md.

The cost models deliberately do not read the prompt n-grams, only hand features
and length. Letting a text model feed them scored the same on dev and made the
spend ratio swing enough to forfeit the balanced tier in 17% of bootstrap
resamples; length alone forfeited none.

The baseline bill is itself unknown at runtime: the cap is a multiple of what
routing everything to ax31-light would cost, and light's token usage is not given
either. It is predicted with the same cost models, which is why the light bill is
estimated rather than measured.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from router_features import episode_text, hand_features

SLACK = 0.03


def main() -> int:
    ap = argparse.ArgumentParser(prog="router-run")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--tier", required=True, choices=("fast", "balanced", "premium"))
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--artifact", type=Path,
                    default=Path(__file__).resolve().parent / "artifact" / "router.pkl")
    args = ap.parse_args()

    with args.artifact.open("rb") as fh:
        art = pickle.load(fh)
    light, upgrades, policy = art["light"], art["upgrades"], art["policy"]
    q_fill, q_safe = art["q_fill"], art["q_safe"]

    blob = json.loads(args.input.read_text(encoding="utf-8"))
    episodes = blob["episodes"] if isinstance(blob, dict) else blob
    ids = [str(e["episode_id"]) for e in episodes]
    texts = [episode_text(e) for e in episodes]
    n = len(ids)
    print(f"{n} episodes, tier {args.tier}", file=sys.stderr, flush=True)

    # Gain reads the first `gain_cap` characters; cost reads no n-grams at all.
    # That split is why this fits the time budget -- the median episode is 237
    # characters and the longest is 71,094, and vectorising that tail was 72% of
    # the runtime.
    capped = [t[: art["gain_cap"]] for t in texts]
    X = hstack([art["tfidf_word"].transform(capped),
                art["tfidf_char"].transform(capped)]).tocsr()
    feats = np.c_[[hand_features(t) for t in texts],
                  np.log1p([len(t) for t in texts])]

    # How many generations the episode is graded over is not given at run time,
    # but it is recoverable from the prompt (dev AUC 0.9990) and it matters: light
    # fails on 38.0% of 2-generation episodes and 18.3% of 4-generation ones.
    Xg = hstack([X, csr_matrix(art["ng_clf"].predict_proba(X)[:, 1][:, None])]).tocsr()

    gain, cost = {}, {q_fill: {}, q_safe: {}}
    for m in upgrades:
        gain[m] = art["gain"][m].predict(Xg)
        for q in (q_fill, q_safe):
            cost[q][m] = art["cost"][q][m].predict(feats).clip(1.0)

    # The cap is a multiple of the all-light bill and the container is never told
    # what that bill is. It is predicted at a low quantile deliberately: over-
    # estimating it inflates the cap and risks an overrun, which forfeits the tier,
    # while under-estimating only leaves some score on the table.
    q_light = art["q_light"]
    light_bill = float(art["light_cost"][q_light].predict(feats).clip(1.0).sum())
    print(f"estimated light bill {light_bill:,.0f} credits "
          f"({light_bill / n:,.0f}/episode, q={q_light})", file=sys.stderr, flush=True)
    cap = float(policy["tiers"][args.tier]["budget_multiplier"]) * light_bill
    target = cap * (1.0 - SLACK)

    picks, spent, taken = [], light_bill, set()
    order = sorted(((gain[m][i] / cost[q_fill][m][i], i, m)
                    for i in range(n) for m in upgrades if gain[m][i] > 0), reverse=True)
    for eff, i, m in order:
        if i in taken:
            continue
        if spent + cost[q_fill][m][i] <= target:
            spent += cost[q_fill][m][i]
            picks.append((eff, i, m))
            taken.add(i)
    while picks and light_bill + sum(cost[q_safe][m][i] for _, i, m in picks) > target:
        picks.pop()

    choice = [light] * n
    for _, i, m in picks:
        choice[i] = m
    counts = {m: choice.count(m) for m in policy["models"]}
    print(f"routed: {counts}", file=sys.stderr, flush=True)

    # submission.v1 requires all six fields. `split` and `tier` are echoed from
    # the run rather than inferred: the operator checks that the submission's
    # challenge_id and split match the input and that the tier matches the run.
    payload = {
        "schema_version": 1,
        "challenge_id": blob.get("challenge_id", "ossp-2026-llm-router-challenge"),
        "policy_id": policy["policy_id"],
        "split": blob.get("split", "unknown"),
        "tier": args.tier,
        "decisions": [{"episode_id": eid, "model_id": mid} for eid, mid in zip(ids, choice)],
    }
    write_atomically(args.output, json.dumps(payload, ensure_ascii=False))
    print(f"wrote {args.output}", file=sys.stderr, flush=True)
    return 0


def write_atomically(path: Path, text: str) -> None:
    """Write `path` via a sibling temp file and one rename.

    RUNTIME.md requires the result to appear atomically -- a partial JSON is not
    accepted as a valid result -- and requires mode 0644. The temp file has to be
    a sibling because rename is only atomic within a filesystem, and the output
    volume is a separate mount. It is unlinked on failure so the volume never
    keeps a file other than submission.json, which the operator checks for.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    # The contract is exit 0 on success and exit 2 -- with a short message, not a
    # traceback -- for input, argument, format or write errors. argparse already
    # exits 2 for bad arguments; this covers everything after it.
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"router-run: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
