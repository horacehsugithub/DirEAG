from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


EPS = 1e-6


def clip01(x: float) -> float:
    return max(EPS, min(1.0 - EPS, float(x)))


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logit(p: float) -> float:
    p = clip01(p)
    return math.log(p / (1.0 - p))


def ece_equal_width(ys: list[float], ps: list[float], bins: int = 10) -> float:
    total = len(ys)
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, p in enumerate(ps) if (lo <= p <= hi if i == 0 else lo < p <= hi)]
        if idx:
            acc = sum(ys[j] for j in idx) / len(idx)
            conf = sum(ps[j] for j in idx) / len(idx)
            out += len(idx) / total * abs(acc - conf)
    return out


def adaptive_ece(ys: list[float], ps: list[float], bins: int = 10) -> float:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = 0.0
    for b in range(bins):
        idx = order[round(b * len(order) / bins): round((b + 1) * len(order) / bins)]
        if idx:
            acc = sum(ys[j] for j in idx) / len(idx)
            conf = sum(ps[j] for j in idx) / len(idx)
            out += len(idx) / len(order) * abs(acc - conf)
    return out


def fast_auroc(ys: list[float], ps: list[float]) -> float | str:
    n_pos = sum(1 for y in ys if y == 1.0)
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return ""
    try:
        return float(roc_auc_score(np.asarray(ys), np.asarray(ps)))
    except Exception:
        return ""


def safe_ap(ys: list[float], ps: list[float]) -> float | str:
    if len(set(ys)) < 2:
        return ""
    return float(average_precision_score(np.asarray(ys), np.asarray(ps)))


def nll(ys: list[float], ps: list[float]) -> float:
    return float(log_loss(np.asarray(ys), np.asarray(ps), labels=[0, 1]))


def metric_summary(rows: list[dict], bins: int = 10) -> dict:
    ys = [1.0 if r["correct"] else 0.0 for r in rows]
    ps = [clip01(r["confidence"]) for r in rows]
    if not rows:
        return {
            "n": 0,
            "accuracy": "",
            "mean_confidence": "",
            "ece": "",
            "adaptive_ece": "",
            "brier": "",
            "nll": "",
            "auroc": "",
            "auprc_positive": "",
            "auprc_negative": "",
        }
    return {
        "n": len(rows),
        "accuracy": sum(ys) / len(ys),
        "mean_confidence": sum(ps) / len(ps),
        "ece": ece_equal_width(ys, ps, bins=bins),
        "adaptive_ece": adaptive_ece(ys, ps, bins=bins),
        "brier": sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps),
        "nll": nll(ys, ps),
        "auroc": fast_auroc(ys, ps),
        "auprc_positive": safe_ap(ys, ps),
        "auprc_negative": safe_ap([1.0 - y for y in ys], [1.0 - p for p in ps]),
    }


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)
