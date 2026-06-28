from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.optimize import minimize

from .data import dataset_targets_for_spec
from .metrics import clip01, logit, metric_summary, sigmoid
from .parse import exact_numeric_match, normalize_numeric_answer
from .utils import ensure_dirs, output_jsonl_path, read_jsonl, resolve_path


LEVELS = 5
NULL_STATE = "__NULL__"


def softplus(x: float) -> float:
    if x > 30:
        return x
    return math.log1p(math.exp(x))


def inv_softplus(y: float) -> float:
    return math.log(math.exp(y) - 1.0)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_records(path: Path) -> list[dict]:
    records = []
    for row in read_jsonl(path):
        outputs = []
        for out in row.get("outputs", []):
            if out.get("pred_answer_norm") is None or out.get("confidence") is None:
                continue
            item = dict(out)
            item["confidence01"] = clip01(float(out["confidence"]) / 100.0)
            item["answer"] = str(out["pred_answer_norm"])
            outputs.append(item)
        outputs.sort(key=lambda x: int(x.get("level_index", 0)))
        if outputs:
            row["usable_outputs"] = outputs
            records.append(row)
    return records


def make_folds(n: int, folds: int, seed: int) -> list[list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    out = [[] for _ in range(folds)]
    for pos, idx in enumerate(indices):
        out[pos % folds].append(idx)
    return out


def correct_answer(pred: str | None, gold: str | None) -> bool:
    return exact_numeric_match(pred, gold)


def majority_selector(record: dict) -> tuple[str, float]:
    outputs = record["usable_outputs"]
    counts = Counter(o["answer"] for o in outputs)
    max_count = max(counts.values())
    tied = [a for a, n in counts.items() if n == max_count]
    if len(tied) == 1:
        return tied[0], max_count / len(outputs)
    means = {
        ans: statistics.mean(o["confidence01"] for o in outputs if o["answer"] == ans)
        for ans in tied
    }
    return max(means, key=means.get), max_count / len(outputs)


def mean_conf_selector(record: dict) -> tuple[str, float]:
    sums = defaultdict(float)
    for o in record["usable_outputs"]:
        sums[o["answer"]] += o["confidence01"]
    answer = max(sums, key=sums.get)
    total = sum(sums.values())
    return answer, sums[answer] / total if total > 0 else 0.0


def steerconf_selector(record: dict) -> tuple[str, float]:
    outputs = record["usable_outputs"]
    confs = [o["confidence01"] for o in outputs]
    mean_conf = statistics.mean(confs)
    std_conf = statistics.pstdev(confs) if len(confs) > 1 else 0.0
    _, ans_cons = majority_selector(record)
    conf_cons = 1.0 / (1.0 + std_conf / mean_conf) if mean_conf > 0 else 1.0
    score = clip01(mean_conf * ans_cons * conf_cons)
    cmin, cmax = min(confs), max(confs)
    if cmax > cmin:
        idx = math.floor(((score - cmin) / (cmax - cmin)) * len(outputs))
        idx = max(0, min(len(outputs) - 1, idx))
    else:
        idx = 0
    return outputs[idx]["answer"], score


def prediction_row(record: dict, method: str, answer: str | None, confidence: float, fold: int | str = "") -> dict:
    return {
        "dataset": record.get("dataset", ""),
        "problem_id": record["problem_id"],
        "method": method,
        "pred_answer_norm": answer,
        "gold_answer_raw": record["gold_answer_raw"],
        "confidence": clip01(confidence),
        "correct": correct_answer(answer, record["gold_answer_raw"]),
        "fold": fold,
    }


def baseline_row(record: dict, method: str, fold: int | str = "") -> dict:
    if method == "majority":
        ans, conf = majority_selector(record)
    elif method == "mean_conf":
        ans, conf = mean_conf_selector(record)
    elif method == "steerconf":
        ans, conf = steerconf_selector(record)
    else:
        raise ValueError(method)
    return prediction_row(record, method, ans, conf, fold=fold)


def fit_platt(rows: list[dict], steps: int = 1200, lr: float = 0.05, l2: float = 0.001) -> tuple[float, float]:
    if len({bool(r["correct"]) for r in rows}) < 2:
        return 0.0, 1.0
    a = 0.0
    b = 1.0
    xs = [logit(r["confidence"]) for r in rows]
    ys = [1.0 if r["correct"] else 0.0 for r in rows]
    for _ in range(steps):
        ga = 0.0
        gb = 0.0
        for x, y in zip(xs, ys):
            p = sigmoid(a + b * x)
            ga += p - y
            gb += (p - y) * x
        ga = ga / len(rows) + l2 * a
        gb = gb / len(rows) + l2 * (b - 1.0)
        a -= lr * ga
        b -= lr * gb
    return a, b


def apply_platt(row: dict, a: float, b: float, method_name: str) -> dict:
    out = dict(row)
    out["method"] = method_name
    out["confidence"] = clip01(sigmoid(a + b * logit(row["confidence"])))
    return out


def unpack(params: list[float], mode: str) -> dict:
    pos = 0
    weights = [softplus(params[pos + i]) for i in range(LEVELS)]
    pos += LEVELS
    intercepts = None
    slope = None
    gamma = 0.0
    if mode in {"level_bias", "qcal"}:
        intercepts = [params[pos + i] for i in range(LEVELS)]
        pos += LEVELS
    if mode == "qcal":
        slope = params[pos]
        pos += 1
    eta = softplus(params[pos])
    pos += 1
    null_base = softplus(params[pos])
    pos += 1
    if mode in {"level_bias", "qcal"}:
        gamma = softplus(params[pos])
    return {
        "weights": weights,
        "intercepts": intercepts,
        "slope": slope,
        "eta": eta,
        "null_base": null_base,
        "gamma": gamma,
    }


def q_value(output: dict, cfg: dict, mode: str) -> float:
    if mode == "count":
        return 1.0
    lv = int(output["level_index"])
    if mode == "level_bias":
        return clip01(sigmoid(cfg["intercepts"][lv]))
    if mode == "qcal":
        return clip01(sigmoid(cfg["intercepts"][lv] + cfg["slope"] * logit(output["confidence01"])))
    raise ValueError(mode)


def posterior(record: dict, cfg: dict, mode: str) -> tuple[dict[str, float], float]:
    outputs = record["usable_outputs"]
    candidates = sorted({o["answer"] for o in outputs})
    prior = cfg["eta"] / (len(candidates) + 1.0)
    alpha = {a: prior for a in candidates}
    alpha_null = prior + cfg["null_base"]
    for o in outputs:
        lv = int(o["level_index"])
        w = cfg["weights"][lv]
        q = q_value(o, cfg, mode)
        alpha[o["answer"]] += w * q
        alpha_null += cfg["gamma"] * w * (1.0 - q)
    total = alpha_null + sum(alpha.values())
    return {a: alpha[a] / total for a in candidates}, alpha_null / total


def target_state(record: dict) -> tuple[str, bool]:
    gold = normalize_numeric_answer(record["gold_answer_raw"])
    for candidate in sorted({str(o["answer"]) for o in record["usable_outputs"]}):
        if normalize_numeric_answer(candidate) == gold:
            return candidate, True
    return NULL_STATE, False


def resolve_null_target_weight(records: list[dict], setting: str | float | int) -> float:
    if isinstance(setting, (float, int)):
        return max(0.0, float(setting))
    text = str(setting).strip().lower()
    if text in {"", "none", "1", "unit", "unweighted"}:
        return 1.0
    if text != "balanced":
        return max(0.0, float(text))
    present = 0
    missing = 0
    for record in records:
        _, in_candidates = target_state(record)
        present += int(in_candidates)
        missing += int(not in_candidates)
    if present == 0 or missing == 0:
        return 1.0
    return present / missing


def categorical_loss(params: list[float], records: list[dict], mode: str, l2: float, null_target_weight: float = 1.0) -> float:
    cfg = unpack(params, mode)
    loss = 0.0
    weight_sum = 0.0
    for record in records:
        probs, p_null = posterior(record, cfg, mode)
        target, in_candidates = target_state(record)
        p_target = probs[target] if in_candidates else p_null
        weight = 1.0 if in_candidates else null_target_weight
        loss -= weight * math.log(max(1e-6, p_target))
        weight_sum += weight
    reg = l2 * sum(x * x for x in params) / max(1, len(params))
    return loss / max(1e-12, weight_sum) + reg


def initial_params(mode: str) -> list[float]:
    params = [inv_softplus(1.0)] * LEVELS
    if mode in {"level_bias", "qcal"}:
        params += [0.0] * LEVELS
    if mode == "qcal":
        params += [0.05]
    params += [inv_softplus(0.1), inv_softplus(0.01)]
    if mode in {"level_bias", "qcal"}:
        params += [inv_softplus(0.25)]
    return params


def bounds(mode: str) -> list[tuple[float, float]]:
    out = [(-5.0, 3.0)] * LEVELS
    if mode in {"level_bias", "qcal"}:
        out += [(-5.0, 5.0)] * LEVELS
    if mode == "qcal":
        out += [(-2.0, 2.0)]
    out += [(-8.0, 3.0), (-8.0, 3.0)]
    if mode in {"level_bias", "qcal"}:
        out += [(-8.0, 3.0)]
    return out


def fit_mle(records: list[dict], mode: str, l2: float, maxiter: int, null_target_weight: str | float | int = 1.0) -> tuple[dict, float, list[float], bool, float]:
    starts = [initial_params(mode)]
    resolved_null_weight = resolve_null_target_weight(records, null_target_weight)
    if mode in {"level_bias", "qcal"}:
        optimistic = initial_params(mode)
        for i, val in enumerate([-0.2, -0.5, -0.1, -0.8, -0.4]):
            optimistic[LEVELS + i] = val
        starts.append(optimistic)
    best = None
    for start in starts:
        result = minimize(
            categorical_loss,
            start,
            args=(records, mode, l2, resolved_null_weight),
            method="L-BFGS-B",
            bounds=bounds(mode),
            options={"maxiter": maxiter, "ftol": 1e-8},
        )
        if best is None or result.fun < best.fun:
            best = result
    return unpack(list(best.x), mode), float(best.fun), [float(x) for x in best.x], bool(best.success), resolved_null_weight


def mle_row(record: dict, cfg: dict, mode: str, method: str, fold: int | str = "") -> dict:
    probs, p_null = posterior(record, cfg, mode)
    answer = max(probs, key=probs.get)
    row = prediction_row(record, method, answer, probs[answer], fold=fold)
    target, in_candidates = target_state(record)
    row["null_probability"] = p_null
    row["num_candidates"] = len(probs)
    row["gold_in_candidates"] = in_candidates
    row["target_state"] = target
    row["target_probability"] = probs[target] if in_candidates else p_null
    return row


def config_to_row(dataset: str, fold: int, method: str, mode: str, cfg: dict, loss: float, raw: list[float], success: bool, null_target_weight: float) -> dict:
    row = {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "mode": mode,
        "cal_categorical_nll": loss,
        "null_target_weight": null_target_weight,
        "success": success,
        "eta": cfg["eta"],
        "null_base": cfg["null_base"],
        "gamma": cfg["gamma"],
        "slope": "" if cfg["slope"] is None else cfg["slope"],
        "raw_params_json": json.dumps(raw),
    }
    for i, w in enumerate(cfg["weights"]):
        row[f"weight_{i}"] = w
    if cfg["intercepts"] is not None:
        for i, a in enumerate(cfg["intercepts"]):
            row[f"intercept_{i}"] = a
    return row


def analyze_dataset(config: dict, dataset_name: str, records: list[dict]) -> tuple[list[dict], list[dict]]:
    folds = make_folds(len(records), int(config["analysis"].get("folds", 5)), int(config["analysis"].get("seed", 0)))
    l2 = float(config["analysis"].get("l2_penalty", 0.001))
    maxiter = int(config["analysis"].get("optimization_maxiter", 500))
    null_target_weight = config["analysis"].get("null_target_weight", "balanced")
    modes = {
        "mle_dirichlet_count": "count",
        "mle_dirichlet_level_bias": "level_bias",
        "mle_dirichlet_qcal": "qcal",
    }
    predictions = []
    configs = []
    for fold_id, test_idx in enumerate(folds):
        test_set = set(test_idx)
        cal_records = [r for i, r in enumerate(records) if i not in test_set]
        test_records = [r for i, r in enumerate(records) if i in test_set]
        platt_params = {}
        fitted = {}
        for method, mode in modes.items():
            cfg, loss, raw_params, success, resolved_null_weight = fit_mle(cal_records, mode, l2=l2, maxiter=maxiter, null_target_weight=null_target_weight)
            fitted[method] = (cfg, mode)
            configs.append(config_to_row(dataset_name, fold_id, method, mode, cfg, loss, raw_params, success, resolved_null_weight))
            platt_params[method] = fit_platt([mle_row(r, cfg, mode, method) for r in cal_records])

        for record in test_records:
            for baseline in ["majority", "mean_conf", "steerconf"]:
                row = baseline_row(record, baseline, fold=fold_id)
                predictions.append(row)
            for method, (cfg, mode) in fitted.items():
                row = mle_row(record, cfg, mode, method, fold=fold_id)
                predictions.append(row)
                a, b = platt_params[method]
                predictions.append(apply_platt(row, a, b, f"{method}_platt"))
    return predictions, configs


def analyze_train_test_dataset(
    config: dict,
    dataset_name: str,
    calibration_records: list[dict],
    evaluation_records: list[dict],
) -> tuple[list[dict], list[dict]]:
    l2 = float(config["analysis"].get("l2_penalty", 0.001))
    maxiter = int(config["analysis"].get("optimization_maxiter", 500))
    null_target_weight = config["analysis"].get("null_target_weight", "balanced")
    modes = {
        "mle_dirichlet_count": "count",
        "mle_dirichlet_level_bias": "level_bias",
        "mle_dirichlet_qcal": "qcal",
    }
    predictions = []
    configs = []
    fold_id = "train_test"
    platt_params = {}
    fitted = {}
    for method, mode in modes.items():
        cfg, loss, raw_params, success, resolved_null_weight = fit_mle(calibration_records, mode, l2=l2, maxiter=maxiter, null_target_weight=null_target_weight)
        fitted[method] = (cfg, mode)
        configs.append(config_to_row(dataset_name, fold_id, method, mode, cfg, loss, raw_params, success, resolved_null_weight))
        platt_params[method] = fit_platt([mle_row(r, cfg, mode, method) for r in calibration_records])

    for record in evaluation_records:
        for baseline in ["majority", "mean_conf", "steerconf"]:
            row = baseline_row(record, baseline, fold=fold_id)
            predictions.append(row)
        for method, (cfg, mode) in fitted.items():
            row = mle_row(record, cfg, mode, method, fold=fold_id)
            predictions.append(row)
            a, b = platt_params[method]
            predictions.append(apply_platt(row, a, b, f"{method}_platt"))
    return predictions, configs


def run_analysis(config: dict) -> dict[str, Path]:
    ensure_dirs(config)
    all_predictions = []
    all_configs = []
    for spec in config["datasets"]:
        dataset_name = str(spec["name"])
        targets = dataset_targets_for_spec(spec)
        if spec.get("calibration_split"):
            target_by_role = {t["role"]: t for t in targets}
            cal_path = output_jsonl_path(config, target_by_role["calibration"]["target_name"])
            eval_path = output_jsonl_path(config, target_by_role["evaluation"]["target_name"])
            calibration_records = load_records(cal_path)
            evaluation_records = load_records(eval_path)
            if not calibration_records or not evaluation_records:
                continue
            preds, cfgs = analyze_train_test_dataset(config, dataset_name, calibration_records, evaluation_records)
        else:
            path = output_jsonl_path(config, targets[0]["target_name"])
            records = load_records(path)
            if not records:
                continue
            preds, cfgs = analyze_dataset(config, dataset_name, records)
        all_predictions.extend(preds)
        all_configs.extend(cfgs)

    bins = int(config["analysis"].get("ece_bins", 10))
    metric_rows = []
    for dataset in sorted({r["dataset"] for r in all_predictions}):
        ds_rows = [r for r in all_predictions if r["dataset"] == dataset]
        for method in sorted({r["method"] for r in ds_rows}):
            metric_rows.append({"dataset": dataset, "method": method, **metric_summary([r for r in ds_rows if r["method"] == method], bins=bins)})
    table_dir = resolve_path(config, "table_dir")
    paths = {
        "predictions": table_dir / "second_experiment_predictions.csv",
        "metrics": table_dir / "second_experiment_metrics.csv",
        "configs": table_dir / "second_experiment_configs.csv",
    }
    write_csv(paths["predictions"], all_predictions)
    write_csv(paths["metrics"], metric_rows)
    write_csv(paths["configs"], all_configs)
    return paths
