# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
"""Feature extraction shared by training and the container.

Kept in one module so the fitted artifact and the runtime cannot drift apart --
a feature-order mismatch between the two would produce a valid-looking submission
built on garbage predictions, with nothing to catch it.
"""

from __future__ import annotations

import re

import numpy as np

CODE = re.compile(r"```|\bdef \b|\bclass \b|\bimport \b|[{};]|</|/>")
MATH = re.compile(r"[=+\-*/^]|\\frac|\\sum|\\int|\$|\b\d+\s*[+\-*/]\s*\d+")
HANGUL = re.compile(r"[가-힣]")
CJK = re.compile(r"[一-鿿぀-ヿ]")


def episode_text(ep: dict) -> str:
    """A single string per episode, whether it arrives as `prompt` or `messages`."""
    if ep.get("prompt"):
        return str(ep["prompt"])
    parts = []
    for msg in ep.get("messages", []) or []:
        parts.append(f"{msg.get('role', '')}: {msg.get('content', '')}")
    return "\n".join(parts)


def hand_features(text: str) -> list[float]:
    n = max(len(text), 1)
    words = text.split()
    digits = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)
    return [
        len(text), len(words), float(np.log1p(len(text))),
        digits / n, upper / n,
        len(HANGUL.findall(text)) / n,
        len(CJK.findall(text)) / n,
        float(bool(CODE.search(text))), len(CODE.findall(text)) / n * 100,
        float(bool(MATH.search(text))), len(MATH.findall(text)) / n * 100,
        text.count("?"), text.count("\n"),
        float("step by step" in text.lower()), float("prove" in text.lower()),
        float("explain" in text.lower()), float("translate" in text.lower()),
        float(np.mean([len(w) for w in words])) if words else 0.0,
    ]
