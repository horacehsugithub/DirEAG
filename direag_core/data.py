from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_dataset

from .parse import normalize_numeric_answer
from .utils import dataset_csv_path, ensure_dirs


GSM8K_ANSWER_RE = re.compile(chr(35) * 4 + r"\s*([^\n]+)")


def extract_gsm8k_answer(answer_text: str) -> str:
    match = GSM8K_ANSWER_RE.search(str(answer_text))
    if not match:
        raise ValueError(f"Could not find GSM8K final answer in: {str(answer_text)[:200]}")
    return match.group(1).strip()


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
            {
                "target_name": f"{name}__calibration",
                "dataset_name": name,
                "role": "calibration",
                "spec": cal_spec,
            },
            {
                "target_name": f"{name}__evaluation",
                "dataset_name": name,
                "role": "evaluation",
                "spec": eval_spec,
            },
        ]
    return [{"target_name": name, "dataset_name": name, "role": "crossfit", "spec": dict(spec)}]


def dataset_targets(config: dict) -> list[dict]:
    targets = []
    for spec in config["datasets"]:
        targets.extend(dataset_targets_for_spec(spec))
    return targets


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
    value = first_existing(
        row,
        [
            "question",
            "Question",
            "input",
            "problem",
            "Problem",
            "sQuestion",
            "original_question",
        ],
    )
    if value is None:
        raise ValueError(f"Could not infer question column for row with columns: {list(row.index)}")
    return str(value).strip()


def infer_gold_answer(row: pd.Series, dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        answer_text = first_existing(row, ["answer", "Answer"])
        return extract_gsm8k_answer(str(answer_text))
    value = first_existing(
        row,
        [
            "answer",
            "Answer",
            "result",
            "result_float",
            "final_ans",
            "final_answer",
            "target",
            "label",
            "correct",
        ],
    )
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        raise ValueError(f"Could not infer answer column for row with columns: {list(row.index)}")
    return str(value).strip()


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


def prepare_all_datasets(config: dict) -> list[Path]:
    ensure_dirs(config)
    return [standardize_target(config, target) for target in dataset_targets(config)]
