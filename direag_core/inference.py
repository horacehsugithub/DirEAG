from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import dataset_targets
from .parse import parse_model_output
from .prompts import STEERING_LEVELS, build_prompt
from .utils import append_jsonl, dataset_csv_path, done_problem_ids, ensure_dirs, output_jsonl_path


def chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_model(config: dict):
    gen_cfg = config["generation"]
    dtype = getattr(torch, str(gen_cfg.get("torch_dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(
        gen_cfg["model_name_or_path"],
        trust_remote_code=bool(gen_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": gen_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(gen_cfg.get("trust_remote_code", True)),
    }
    if bool(gen_cfg.get("load_in_4bit", False)):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=str(gen_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(gen_cfg.get("bnb_4bit_use_double_quant", True)),
        )
    model = AutoModelForCausalLM.from_pretrained(gen_cfg["model_name_or_path"], **model_kwargs)
    model.eval()
    return tokenizer, model


def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def build_format_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Your previous response may not have followed the required output format. "
        "Answer again and make the final line exactly:\n"
        "Answer and Confidence (0-100): <final numeric answer>, <confidence 0-100>%\n"
        "Do not put brackets around either number."
    )


def generate_batch(tokenizer, model, prompts: list[str], config: dict) -> list[str]:
    gen_cfg = config["generation"]
    texts = [render_chat(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    temperature = float(gen_cfg.get("temperature", 0.0))
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


def generate_batch_with_fallback(tokenizer, model, prompts: list[str], config: dict) -> list[str]:
    try:
        return generate_batch(tokenizer, model, prompts, config)
    except RuntimeError as exc:
        is_cuda_oom = "CUDA out of memory" in str(exc) or "CUBLAS_STATUS_ALLOC_FAILED" in str(exc)
        if not is_cuda_oom or len(prompts) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(prompts) // 2
        left = generate_batch_with_fallback(tokenizer, model, prompts[:midpoint], config)
        right = generate_batch_with_fallback(tokenizer, model, prompts[midpoint:], config)
        return left + right


def infer_dataset(config: dict, target: dict, tokenizer, model) -> Path:
    target_name = target["target_name"]
    dataset_name = target["dataset_name"]
    sample_path = dataset_csv_path(config, target_name)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing standardized data: {sample_path}")
    out_path = output_jsonl_path(config, target_name)
    finished = done_problem_ids(out_path)
    sample = pd.read_csv(sample_path, dtype=str)
    pending = [row for _, row in sample.iterrows() if str(row["problem_id"]) not in finished]
    limit = config["generation"].get("runtime_limit_per_dataset")
    if limit is not None:
        pending = pending[: max(0, int(limit))]

    use_cot = bool(config["generation"].get("use_cot", True))
    problem_batch_size = max(1, int(config["generation"].get("problem_batch_size", 1)))
    prompt_batch_size = max(1, int(config["generation"].get("prompt_batch_size", problem_batch_size * len(STEERING_LEVELS))))
    progress_total = len(finished) + len(pending) if limit is not None else len(sample)

    with tqdm(total=progress_total, initial=len(finished), desc=f"{target_name} steered inference") as progress:
        for batch in chunks(pending, problem_batch_size):
            prompt_items = []
            for row in batch:
                for level in STEERING_LEVELS:
                    prompt_items.append((row, level, build_prompt(str(row["question"]), level, use_cot=use_cot)))

            raw_outputs = []
            for prompt_batch in chunks(prompt_items, prompt_batch_size):
                raw_outputs.extend(generate_batch_with_fallback(tokenizer, model, [item[2] for item in prompt_batch], config))

            parse_retries = max(0, int(config["generation"].get("parse_retries", 0)))
            for _ in range(parse_retries):
                parsed = [parse_model_output(raw) for raw in raw_outputs]
                failed_indices = [i for i, parsed_item in enumerate(parsed) if not parsed_item["parse_success"]]
                if not failed_indices:
                    break
                retry_prompts = [build_format_retry_prompt(prompt_items[i][2]) for i in failed_indices]
                retry_outputs = []
                for retry_batch in chunks(retry_prompts, prompt_batch_size):
                    retry_outputs.extend(generate_batch_with_fallback(tokenizer, model, retry_batch, config))
                for idx, retry_output in zip(failed_indices, retry_outputs, strict=True):
                    raw_outputs[idx] = retry_output

            outputs_by_problem = {str(row["problem_id"]): [] for row in batch}
            for (row, level, _prompt), raw in zip(prompt_items, raw_outputs, strict=True):
                outputs_by_problem[str(row["problem_id"])].append(
                    {
                        "level_index": level["index"],
                        "level": level["level"],
                        "level_name": level["name"],
                        "raw_output": raw,
                        **parse_model_output(raw),
                    }
                )

            for row in batch:
                problem_id = str(row["problem_id"])
                append_jsonl(
                    out_path,
                    {
                        "dataset": dataset_name,
                        "split_role": row.get("split_role", target["role"]),
                        "source_split": row.get("source_split", target["spec"].get("split", "")),
                        "problem_id": problem_id,
                        "question": row["question"],
                        "gold_answer_raw": row["gold_answer_raw"],
                        "gold_answer_norm": row.get("gold_answer_norm"),
                        "outputs": sorted(outputs_by_problem[problem_id], key=lambda x: int(x["level_index"])),
                    },
                )
                finished.add(problem_id)
                progress.update(1)
    return out_path


def run_inference(config: dict) -> list[Path]:
    ensure_dirs(config)
    tokenizer, model = load_model(config)
    return [infer_dataset(config, target, tokenizer, model) for target in dataset_targets(config)]
