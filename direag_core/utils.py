from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def repo_root_from_config(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(config.get("project_root", "."))
    if root.is_absolute():
        return root
    return (config_path.parent.parent / root).resolve() if root.name != "second_experiment" else config_path.parents[1].resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    config["_root"] = str(repo_root_from_config(config_path, config))
    return config


def root(config: dict[str, Any]) -> Path:
    return Path(config["_root"])


def resolve_path(config: dict[str, Any], key: str) -> Path:
    rel = config["paths"][key]
    path = Path(rel)
    return path if path.is_absolute() else root(config) / path


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ["data_dir", "standardized_data_dir", "output_dir", "steered_output_dir", "table_dir", "doc_dir"]:
        resolve_path(config, key).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    return {str(row.get("problem_id")) for row in read_jsonl(path)}


def dataset_csv_path(config: dict[str, Any], dataset_name: str) -> Path:
    return resolve_path(config, "standardized_data_dir") / f"{dataset_name}.csv"


def output_jsonl_path(config: dict[str, Any], dataset_name: str) -> Path:
    model_tag = str(config["generation"].get("model_tag", "model"))
    return resolve_path(config, "steered_output_dir") / model_tag / f"{dataset_name}.jsonl"
