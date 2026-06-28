from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


ANSWER_LINE_RE = re.compile(
    r"Answer\s+and\s+Confidence\s*\(0\s*-\s*100\)\s*:\s*(?P<body>[^\r\n]*)",
    flags=re.IGNORECASE | re.DOTALL,
)
CONF_RE = re.compile(r"(?P<conf>\d+(?:\.\d+)?)\s*%")
SCI_SUFFIX = r"(?:[eE][-+]?\d+)?"
NUMBER_PATTERN = r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUM_RE = re.compile(rf"{NUMBER_PATTERN}(?:\s*/\s*{NUMBER_PATTERN})?")
EQUALS_RESULT_RE = re.compile(rf"=\s*(?P<num>\$?{NUMBER_PATTERN}(?:\s*/\s*{NUMBER_PATTERN})?)")


def clean_answer_conf_body(body: str) -> str:
    return body.strip().strip("`").replace("[", "").replace("]", "").strip()


def extract_confidence(text: str) -> float | None:
    match = NUM_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return max(0.0, min(100.0, float(match.group(0).replace(" ", ""))))
    except ValueError:
        return None


def extract_last_calculation_result(text: str) -> str | None:
    matches = EQUALS_RESULT_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> str | None:
    matches = NUM_RE.findall(text.replace(",", ""))
    return matches[-1].strip() if matches else None


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


def parse_model_output(text: str) -> dict:
    match = ANSWER_LINE_RE.search(text)
    if match:
        body = clean_answer_conf_body(match.group("body"))
        if "," in body:
            answer, conf_text = [part.strip() for part in body.rsplit(",", 1)]
            confidence = extract_confidence(conf_text)
            answer_norm = normalize_numeric_answer(answer)
            return {
                "pred_answer_raw": answer,
                "pred_answer_norm": answer_norm,
                "confidence": confidence,
                "parse_success": answer_norm is not None and confidence is not None,
            }

        single_conf = extract_confidence(body)
        recovered_answer = extract_last_calculation_result(text[: match.start()])
        recovered_norm = normalize_numeric_answer(recovered_answer)
        return {
            "pred_answer_raw": "" if recovered_answer is None else recovered_answer,
            "pred_answer_norm": recovered_norm,
            "confidence": single_conf,
            "parse_success": recovered_norm is not None and single_conf is not None,
        }

    confidence = None
    conf_match = CONF_RE.search(text)
    if conf_match:
        confidence = max(0.0, min(100.0, float(conf_match.group("conf"))))
    answer = extract_last_number(text)
    return {
        "pred_answer_raw": "" if answer is None else answer,
        "pred_answer_norm": normalize_numeric_answer(answer) if answer is not None else None,
        "confidence": confidence,
        "parse_success": answer is not None and confidence is not None,
    }
