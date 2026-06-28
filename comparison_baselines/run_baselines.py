from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import DatasetDict, load_dataset
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


EPS = 1e-6
NUMBER_PATTERN = r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUM_RE = re.compile(rf"{NUMBER_PATTERN}(?:\s*/\s*{NUMBER_PATTERN})?")
GSM8K_ANSWER_RE = re.compile(chr(35) * 4 + r"\s*([^\n]+)")
ANSWER_LINE_RE = re.compile(r"Final\s+Answer\s*:\s*(?P<body>[^\r\n]+)", flags=re.IGNORECASE)
TOPK_LINE_RE = re.compile(
    rf"(?:^|\n)\s*(?:[-*]|\d+[\).:-])?\s*(?:answer\s*)?(?P<answer>{NUMBER_PATTERN}(?:\s*/\s*{NUMBER_PATTERN})?)"
    rf".*?(?P<conf>\d+(?:\.\d+)?)\s*%",
    flags=re.IGNORECASE,
)
ENV_DEFAULT_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-?(?P<default>[^}]*)\}")


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name = match.group("name")
            default = match.group("default")
            return os.environ.get(name, default)

        return os.path.expandvars(ENV_DEFAULT_RE.sub(repl, value))
    if isinstance(value, list):
        return [expand_env_vars(x) for x in value]
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = expand_env_vars(yaml.safe_load(f))
    root = Path(config.get("project_root", "."))
    if not root.is_absolute():
        root = (path.parent.parent / root).resolve()
    config["_root"] = str(root)
    config["_config_path"] = str(path)
    return config


def root(config: dict) -> Path:
    return Path(config["_root"])


def resolve_path(config: dict, key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else root(config) / path


def ensure_dirs(config: dict) -> None:
    for key in ["data_dir", "standardized_data_dir", "output_dir", "generation_dir", "table_dir", "doc_dir"]:
        resolve_path(config, key).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def done_problem_ids(path: Path) -> set[str]:
    return {str(r.get("problem_id")) for r in read_jsonl(path)}


def chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def clip01(x: float) -> float:
    return max(EPS, min(1.0 - EPS, float(x)))


def normalize_numeric_answer(value: str | int | float | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "").strip()
    if text.endswith("."):
        text = text[:-1]
    if not text:
        return None
    match = NUM_RE.search(text)
    if not match:
        return None
    token = match.group(0).replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            dec = Decimal(numerator) / Decimal(denominator)
        else:
            dec = Decimal(token)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return str(dec.normalize())


def exact_numeric_match(pred_norm: str | None, gold_raw: str | int | float | None) -> bool:
    gold_norm = normalize_numeric_answer(gold_raw)
    return pred_norm is not None and gold_norm is not None and pred_norm == gold_norm


def extract_last_number(text: str) -> str | None:
    matches = NUM_RE.findall(str(text).replace(",", ""))
    return matches[-1].strip() if matches else None


def parse_final_answer(text: str) -> dict:
    match = ANSWER_LINE_RE.search(str(text))
    answer = match.group("body").strip() if match else extract_last_number(str(text))
    answer_norm = normalize_numeric_answer(answer)
    return {
        "pred_answer_raw": "" if answer is None else answer,
        "pred_answer_norm": answer_norm,
        "parse_success": answer_norm is not None,
    }


def parse_topk_output(text: str, k: int) -> dict:
    items = []
    seen_spans = set()
    for match in TOPK_LINE_RE.finditer(str(text)):
        if match.span() in seen_spans:
            continue
        seen_spans.add(match.span())
        ans_raw = match.group("answer")
        ans_norm = normalize_numeric_answer(ans_raw)
        if ans_norm is None:
            continue
        conf = max(0.0, min(100.0, float(match.group("conf"))))
        items.append({"answer_raw": ans_raw, "answer_norm": ans_norm, "confidence": conf})
        if len(items) >= k:
            break
    if not items:
        fallback = parse_final_answer(text)
        return {
            "topk_items": [],
            "pred_answer_raw": fallback["pred_answer_raw"],
            "pred_answer_norm": fallback["pred_answer_norm"],
            "confidence": None,
            "parse_success": False,
        }
    best = max(items, key=lambda x: x["confidence"])
    return {
        "topk_items": items,
        "pred_answer_raw": best["answer_raw"],
        "pred_answer_norm": best["answer_norm"],
        "confidence": best["confidence"],
        "parse_success": best["answer_norm"] is not None and best["confidence"] is not None,
    }


def get_split(dataset_obj: Any, split: str):
    if isinstance(dataset_obj, DatasetDict):
        if split in dataset_obj:
            return dataset_obj[split]
        if "test" in dataset_obj:
            return dataset_obj["test"]
        if "validation" in dataset_obj:
            return dataset_obj["validation"]
        return dataset_obj[next(iter(dataset_obj.keys()))]
    return dataset_obj


def load_hf_dataset(spec: dict):
    hf_config = spec.get("hf_config")
    split = spec.get("split")
    if hf_config in (None, "null", ""):
        try:
            return load_dataset(spec["hf_name"], split=split)
        except Exception:
            dataset_obj = load_dataset(spec["hf_name"])
            return get_split(dataset_obj, split or "test")
    return load_dataset(spec["hf_name"], hf_config, split=split)


def dataset_targets_for_spec(spec: dict) -> list[dict]:
    name = str(spec["name"])
    eval_split = str(spec.get("evaluation_split", spec.get("split", "test")))
    if spec.get("calibration_split"):
        cal_spec = dict(spec)
        cal_spec["split"] = spec["calibration_split"]
        if "calibration_sample_size" in spec:
            cal_spec["sample_size"] = spec["calibration_sample_size"]
        eval_spec = dict(spec)
        eval_spec["split"] = eval_split
        if "evaluation_sample_size" in spec:
            eval_spec["sample_size"] = spec["evaluation_sample_size"]
        return [
            {"target_name": f"{name}__calibration", "dataset_name": name, "role": "calibration", "spec": cal_spec},
            {"target_name": f"{name}__evaluation", "dataset_name": name, "role": "evaluation", "spec": eval_spec},
        ]
    return [{"target_name": name, "dataset_name": name, "role": "crossfit", "spec": dict(spec)}]


def dataset_targets(config: dict) -> list[dict]:
    out = []
    for spec in config["datasets"]:
        out.extend(dataset_targets_for_spec(spec))
    return out


def first_existing(row: pd.Series, candidates: list[str]) -> Any:
    lower_to_col = {str(c).lower(): c for c in row.index}
    for name in candidates:
        col = lower_to_col.get(name.lower())
        if col is not None and pd.notna(row[col]):
            return row[col]
    return None


def infer_question(row: pd.Series, dataset_name: str) -> str:
    if dataset_name == "svamp":
        body = first_existing(row, ["Body", "body"])
        question = first_existing(row, ["Question", "question"])
        if body is not None and question is not None:
            return f"{body} {question}".strip()
    value = first_existing(row, ["question", "Question", "input", "problem", "Problem", "sQuestion", "original_question"])
    if value is None:
        raise ValueError(f"Could not infer question column for row with columns: {list(row.index)}")
    return str(value).strip()


def extract_gsm8k_answer(answer_text: str) -> str:
    match = GSM8K_ANSWER_RE.search(str(answer_text))
    if not match:
        raise ValueError(f"Could not find GSM8K final answer in: {str(answer_text)[:200]}")
    return match.group(1).strip()


def infer_gold_answer(row: pd.Series, dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        answer_text = first_existing(row, ["answer", "Answer"])
        return extract_gsm8k_answer(str(answer_text))
    value = first_existing(row, ["answer", "Answer", "result", "result_float", "final_ans", "final_answer", "target", "label", "correct"])
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        raise ValueError(f"Could not infer answer column for row with columns: {list(row.index)}")
    return str(value).strip()


def dataset_csv_path(config: dict, target_name: str) -> Path:
    return resolve_path(config, "standardized_data_dir") / f"{target_name}.csv"


def standardize_target(config: dict, target: dict) -> Path:
    dataset_name = target["dataset_name"]
    spec = target["spec"]
    dataset = load_hf_dataset(spec)
    df = pd.DataFrame(dataset)
    rows = []
    for i, row in df.iterrows():
        question = infer_question(row, dataset_name)
        gold_answer = infer_gold_answer(row, dataset_name)
        gold_norm = normalize_numeric_answer(gold_answer)
        if gold_norm is None:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "split_role": target["role"],
                "source_split": spec.get("split", "default"),
                "problem_id": f"{dataset_name}::{target['role']}::{spec.get('split', 'default')}::{i:05d}",
                "question": question,
                "gold_answer_raw": gold_answer,
                "gold_answer_norm": gold_norm,
            }
        )
    out = pd.DataFrame(rows)
    sample_size = spec.get("sample_size")
    if sample_size and int(sample_size) < len(out):
        out = out.sample(n=int(sample_size), random_state=int(spec.get("sample_seed", 0))).reset_index(drop=True)
    path = dataset_csv_path(config, target["target_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def prepare_data(config: dict) -> list[Path]:
    ensure_dirs(config)
    return [standardize_target(config, target) for target in dataset_targets(config)]


def build_vanilla_sample_prompt(question: str) -> str:
    return (
        "Read the question, analyze step by step, and provide the final numeric answer.\n"
        "Use exactly this format at the end:\n"
        "Final Answer: [ONLY the final numeric answer]\n\n"
        f"Question:\n{question}"
    )


def build_topk_prompt(question: str, k: int) -> str:
    return (
        f"Read the question and analyze step by step. Then provide your top {k} most likely final numeric answers "
        "and your confidence for each answer.\n"
        "The confidences should be percentages from 0 to 100.\n"
        "Use exactly this final format:\n"
        "Top Answers:\n"
        "1. [numeric answer], [confidence]%\n"
        "2. [numeric answer], [confidence]%\n"
        "...\n\n"
        f"Question:\n{question}"
    )


def load_model(model_cfg: dict):
    dtype = getattr(torch, str(model_cfg.get("torch_dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = {
        "torch_dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    if bool(model_cfg.get("load_in_4bit", False)):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        )
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name_or_path"], **kwargs)
    model.eval()
    return tokenizer, model


def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_batch(tokenizer, model, prompts: list[str], gen_cfg: dict) -> list[str]:
    texts = [render_chat(tokenizer, p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    temperature = float(gen_cfg.get("temperature", 0.7))
    generation_kwargs = {
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 1024)),
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = float(gen_cfg.get("top_p", 1.0))
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    input_width = inputs["input_ids"].shape[-1]
    return [tokenizer.decode(row[input_width:], skip_special_tokens=True) for row in generated]


def generate_batch_with_fallback(tokenizer, model, prompts: list[str], gen_cfg: dict) -> list[str]:
    try:
        return generate_batch(tokenizer, model, prompts, gen_cfg)
    except RuntimeError as exc:
        is_cuda_oom = "CUDA out of memory" in str(exc) or "CUBLAS_STATUS_ALLOC_FAILED" in str(exc)
        if not is_cuda_oom or len(prompts) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(prompts) // 2
        return generate_batch_with_fallback(tokenizer, model, prompts[:mid], gen_cfg) + generate_batch_with_fallback(tokenizer, model, prompts[mid:], gen_cfg)


def generation_path(config: dict, model_tag: str, target_name: str, kind: str) -> Path:
    return resolve_path(config, "generation_dir") / model_tag / f"{target_name}__{kind}.jsonl"


def infer_target(config: dict, model_cfg: dict, target: dict, tokenizer, model) -> list[Path]:
    gen_cfg = config["generation"]
    model_tag = model_cfg["model_tag"]
    target_name = target["target_name"]
    sample_path = dataset_csv_path(config, target_name)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing standardized data: {sample_path}")
    sample = pd.read_csv(sample_path, dtype=str)
    limit = gen_cfg.get("runtime_limit_per_dataset")
    if limit is not None:
        sample = sample.head(int(limit)).copy()

    paths = []
    if bool(gen_cfg.get("run_sample_baselines", True)):
        paths.append(infer_sample_file(config, model_tag, target, sample, tokenizer, model))
    if bool(gen_cfg.get("run_topk_baseline", True)):
        paths.append(infer_topk_file(config, model_tag, target, sample, tokenizer, model))
    return paths


def infer_sample_file(config: dict, model_tag: str, target: dict, sample: pd.DataFrame, tokenizer, model) -> Path:
    gen_cfg = config["generation"]
    out_path = generation_path(config, model_tag, target["target_name"], "sample")
    finished = done_problem_ids(out_path)
    k = int(gen_cfg.get("sample_k", 5))
    pending = [row for _, row in sample.iterrows() if str(row["problem_id"]) not in finished]
    prompt_batch_size = max(1, int(gen_cfg.get("prompt_batch_size", 10)))
    problem_batch_size = max(1, int(gen_cfg.get("problem_batch_size", 2)))
    with tqdm(total=len(finished) + len(pending), initial=len(finished), desc=f"{model_tag}/{target['target_name']} sample") as progress:
        for batch in chunks(pending, problem_batch_size):
            prompt_items = []
            for row in batch:
                prompt = build_vanilla_sample_prompt(str(row["question"]))
                for sample_id in range(k):
                    prompt_items.append((row, sample_id, prompt))
            raw_outputs = []
            for prompt_batch in chunks(prompt_items, prompt_batch_size):
                raw_outputs.extend(generate_batch_with_fallback(tokenizer, model, [item[2] for item in prompt_batch], gen_cfg))
            by_problem = defaultdict(list)
            for (row, sample_id, _), raw in zip(prompt_items, raw_outputs, strict=True):
                by_problem[str(row["problem_id"])].append({"sample_id": sample_id, "raw_output": raw, **parse_final_answer(raw)})
            for row in batch:
                pid = str(row["problem_id"])
                append_jsonl(
                    out_path,
                    {
                        "model_tag": model_tag,
                        "dataset": row["dataset"],
                        "split_role": row.get("split_role", target["role"]),
                        "source_split": row.get("source_split", ""),
                        "problem_id": pid,
                        "question": row["question"],
                        "gold_answer_raw": row["gold_answer_raw"],
                        "gold_answer_norm": row.get("gold_answer_norm"),
                        "samples": sorted(by_problem[pid], key=lambda x: int(x["sample_id"])),
                    },
                )
                progress.update(1)
    return out_path


def infer_topk_file(config: dict, model_tag: str, target: dict, sample: pd.DataFrame, tokenizer, model) -> Path:
    gen_cfg = config["generation"]
    out_path = generation_path(config, model_tag, target["target_name"], "topk")
    finished = done_problem_ids(out_path)
    k = int(gen_cfg.get("topk_k", 5))
    pending = [row for _, row in sample.iterrows() if str(row["problem_id"]) not in finished]
    prompt_batch_size = max(1, int(gen_cfg.get("prompt_batch_size", 10)))
    problem_batch_size = max(1, int(gen_cfg.get("problem_batch_size", 2)))
    with tqdm(total=len(finished) + len(pending), initial=len(finished), desc=f"{model_tag}/{target['target_name']} topk") as progress:
        for batch in chunks(pending, problem_batch_size):
            prompt_items = [(row, build_topk_prompt(str(row["question"]), k)) for row in batch]
            raw_outputs = []
            for prompt_batch in chunks(prompt_items, prompt_batch_size):
                raw_outputs.extend(generate_batch_with_fallback(tokenizer, model, [item[1] for item in prompt_batch], gen_cfg))
            for (row, _), raw in zip(prompt_items, raw_outputs, strict=True):
                append_jsonl(
                    out_path,
                    {
                        "model_tag": model_tag,
                        "dataset": row["dataset"],
                        "split_role": row.get("split_role", target["role"]),
                        "source_split": row.get("source_split", ""),
                        "problem_id": str(row["problem_id"]),
                        "question": row["question"],
                        "gold_answer_raw": row["gold_answer_raw"],
                        "gold_answer_norm": row.get("gold_answer_norm"),
                        "raw_output": raw,
                        **parse_topk_output(raw, k),
                    },
                )
                progress.update(1)
    return out_path


def run_inference(config: dict) -> list[Path]:
    ensure_dirs(config)
    written = []
    targets = dataset_targets(config)
    for model_cfg in config["models"]:
        tokenizer, model = load_model(model_cfg)
        for target in targets:
            written.extend(infer_target(config, model_cfg, target, tokenizer, model))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


def prediction_row(record: dict, model_tag: str, method: str, answer: str | None, confidence: float, fold: int | str = "") -> dict:
    return {
        "model_tag": model_tag,
        "dataset": record.get("dataset", ""),
        "problem_id": record["problem_id"],
        "method": method,
        "pred_answer_norm": answer,
        "gold_answer_raw": record["gold_answer_raw"],
        "confidence": clip01(confidence),
        "correct": exact_numeric_match(answer, record["gold_answer_raw"]),
        "fold": fold,
    }


def majority_answer(items: list[dict]) -> tuple[str | None, float, dict[str, int]]:
    answers = [str(x["pred_answer_norm"]) for x in items if x.get("pred_answer_norm") is not None]
    if not answers:
        return None, 0.0, {}
    counts = Counter(answers)
    max_count = max(counts.values())
    tied = sorted([a for a, n in counts.items() if n == max_count], key=str)
    return tied[0], max_count / len(items), dict(counts)


def sample_baseline_rows(record: dict, model_tag: str, fold: int | str = "") -> list[dict]:
    samples = record.get("samples", [])
    answer, sc_conf, counts = majority_answer(samples)
    if answer is None:
        return []
    k = len(samples)
    probs = [n / k for n in counts.values()] if k else []
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(k) if k > 1 else 1.0
    sem_conf = 1.0 - entropy / max_entropy if max_entropy > 0 else 1.0
    sem_conf = max(0.0, min(1.0, sem_conf))
    return [
        prediction_row(record, model_tag, "sample_consistency", answer, sc_conf, fold),
        prediction_row(record, model_tag, "semantic_entropy", answer, sem_conf, fold),
    ]


def topk_baseline_row(record: dict, model_tag: str, fold: int | str = "") -> list[dict]:
    if record.get("pred_answer_norm") is None or record.get("confidence") is None:
        return []
    return [prediction_row(record, model_tag, "topk", record["pred_answer_norm"], float(record["confidence"]) / 100.0, fold)]


def make_folds(n: int, folds: int, seed: int) -> list[list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    out = [[] for _ in range(folds)]
    for pos, i in enumerate(idx):
        out[pos % folds].append(i)
    return out


def rows_for_records(sample_records: list[dict], topk_records: list[dict], model_tag: str, fold: int | str = "") -> list[dict]:
    rows = []
    for record in sample_records:
        rows.extend(sample_baseline_rows(record, model_tag, fold))
    for record in topk_records:
        rows.extend(topk_baseline_row(record, model_tag, fold))
    return rows


def load_target_records(config: dict, model_tag: str, target_name: str) -> tuple[list[dict], list[dict]]:
    return (
        read_jsonl(generation_path(config, model_tag, target_name, "sample")),
        read_jsonl(generation_path(config, model_tag, target_name, "topk")),
    )


def analyze_train_test(config: dict, model_tag: str, dataset_name: str, cal_target: str, eval_target: str) -> list[dict]:
    eval_sample, eval_topk = load_target_records(config, model_tag, eval_target)
    eval_rows = rows_for_records(eval_sample, eval_topk, model_tag, fold="evaluation")
    return list(eval_rows)


def analyze_crossfit(config: dict, model_tag: str, target_name: str) -> list[dict]:
    seed = int(config["analysis"].get("seed", 0))
    folds_n = int(config["analysis"].get("folds", 5))
    sample_records, topk_records = load_target_records(config, model_tag, target_name)
    n = min(len(sample_records), len(topk_records)) if sample_records and topk_records else max(len(sample_records), len(topk_records))
    folds = make_folds(n, folds_n, seed)
    out = []
    for fold_id, test_idx in enumerate(folds):
        test_set = set(test_idx)
        test_sample = [r for i, r in enumerate(sample_records) if i in test_set]
        test_topk = [r for i, r in enumerate(topk_records) if i in test_set]
        test_rows = rows_for_records(test_sample, test_topk, model_tag, fold=fold_id)
        out.extend(test_rows)
    return out


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


def metric_summary(rows: list[dict], bins: int = 10) -> dict:
    ys = [1.0 if r["correct"] else 0.0 for r in rows]
    ps = [clip01(float(r["confidence"])) for r in rows]
    if not rows:
        return {"n": 0, "accuracy": "", "mean_confidence": "", "ece": "", "brier": "", "nll": "", "auroc": "", "auprc_positive": "", "auprc_negative": ""}
    out = {
        "n": len(rows),
        "accuracy": sum(ys) / len(ys),
        "mean_confidence": sum(ps) / len(ps),
        "ece": ece_equal_width(ys, ps, bins=bins),
        "brier": sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps),
        "nll": float(log_loss(np.asarray(ys), np.asarray(ps), labels=[0, 1])),
    }
    if len(set(ys)) >= 2:
        out["auroc"] = float(roc_auc_score(np.asarray(ys), np.asarray(ps)))
        out["auprc_positive"] = float(average_precision_score(np.asarray(ys), np.asarray(ps)))
        out["auprc_negative"] = float(average_precision_score(np.asarray([1.0 - y for y in ys]), np.asarray([1.0 - p for p in ps])))
    else:
        out["auroc"] = out["auprc_positive"] = out["auprc_negative"] = ""
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(config: dict) -> dict[str, Path]:
    all_predictions = []
    for model_cfg in config["models"]:
        model_tag = model_cfg["model_tag"]
        for spec in config["datasets"]:
            dataset_name = spec["name"]
            targets = dataset_targets_for_spec(spec)
            if spec.get("calibration_split"):
                by_role = {t["role"]: t for t in targets}
                all_predictions.extend(analyze_train_test(config, model_tag, dataset_name, by_role["calibration"]["target_name"], by_role["evaluation"]["target_name"]))
            else:
                all_predictions.extend(analyze_crossfit(config, model_tag, targets[0]["target_name"]))

    bins = int(config["analysis"].get("ece_bins", 10))
    metrics = []
    for model_tag in sorted({r["model_tag"] for r in all_predictions}):
        model_rows = [r for r in all_predictions if r["model_tag"] == model_tag]
        for dataset in sorted({r["dataset"] for r in model_rows}):
            ds_rows = [r for r in model_rows if r["dataset"] == dataset]
            for method in sorted({r["method"] for r in ds_rows}):
                rows = [r for r in ds_rows if r["method"] == method]
                metrics.append({"model_tag": model_tag, "dataset": dataset, "method": method, **metric_summary(rows, bins=bins)})

    table_dir = resolve_path(config, "table_dir")
    paths = {
        "predictions": table_dir / "baseline_predictions.csv",
        "metrics": table_dir / "baseline_metrics.csv",
    }
    write_csv(paths["predictions"], all_predictions)
    write_csv(paths["metrics"], metrics)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top-K, sample consistency, and semantic entropy baselines.")
    parser.add_argument("command", choices=["prepare-data", "infer", "analyze", "all"])
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command in {"prepare-data", "all"}:
        for path in prepare_data(config):
            print(f"prepared {path}")
    if args.command in {"infer", "all"}:
        for path in run_inference(config):
            print(f"wrote {path}")
    if args.command in {"analyze", "all"}:
        for name, path in analyze(config).items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
