from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parents[1]

METRIC_SOURCES = [
    {
        "model": "Qwen2.5-7B",
        "dataset": "GSM8K",
        "path": ROOT / "results_gsm8kfull" / "outputs" / "tables" / "second_experiment_metrics.csv",
        "dataset_filter": None,
    },
    {
        "model": "Qwen2.5-7B",
        "dataset": "SVAMP",
        "path": ROOT / "20260518" / "accepted_results" / "svamp" / "metrics.csv",
        "dataset_filter": None,
    },
    {
        "model": "Qwen2.5-7B",
        "dataset": "GSM-Hard",
        "path": ROOT / "20260518" / "accepted_results" / "gsmhard" / "metrics.csv",
        "dataset_filter": None,
    },
    {
        "model": "Mistral-7B",
        "dataset": "GSM8K",
        "path": ROOT / "20260524_result" / "mistral_dirichlet_experiment" / "outputs" / "tables" / "second_experiment_metrics.csv",
        "dataset_filter": "gsm8k",
    },
    {
        "model": "Mistral-7B",
        "dataset": "SVAMP",
        "path": ROOT / "20260524_result" / "mistral_dirichlet_experiment" / "outputs" / "tables" / "second_experiment_metrics.csv",
        "dataset_filter": "svamp",
    },
    {
        "model": "Mistral-7B",
        "dataset": "GSM-Hard",
        "path": ROOT / "20260524_result" / "mistral_dirichlet_experiment" / "outputs" / "tables" / "second_experiment_metrics.csv",
        "dataset_filter": "gsmhard",
    },
]

METHODS = [
    (
        "mle_dirichlet_count_platt",
        "Count-only + Cal.",
        "answer-count evidence with final binary calibration",
    ),
    (
        "mle_dirichlet_level_bias_platt",
        "+ Level Reliability",
        "adds learned steering-level reliability",
    ),
    (
        "mle_dirichlet_qcal",
        "+ Confidence Evidence",
        "adds calibrated verbalized confidence evidence without final binary calibration",
    ),
    (
        "mle_dirichlet_qcal_platt",
        "Full DirEAG",
        "main DirEAG variant with final binary calibration",
    ),
]

METRICS = ["accuracy", "ece", "brier", "auroc", "auprc_positive", "auprc_negative"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return value


def source_rows(source: dict) -> list[dict[str, str]]:
    rows = read_csv(source["path"])
    if source["dataset_filter"] is not None:
        rows = [r for r in rows if r.get("dataset") == source["dataset_filter"]]
    return rows


def build_rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for source in METRIC_SOURCES:
        rows = source_rows(source)
        by_method: dict[str, dict[str, str]] = {}
        for row in rows:
            by_method.setdefault(row["method"], row)
        for method, display, meaning in METHODS:
            if method not in by_method:
                raise RuntimeError(f"Missing {method} in {source['path']} ({source['dataset']})")
            src = by_method[method]
            out_row = {
                "model": source["model"],
                "dataset": source["dataset"],
                "method": method,
                "display_name": display,
                "ablation_meaning": meaning,
            }
            for metric in METRICS:
                out_row[metric] = fmt(src.get(metric, ""))
            out.append(out_row)
    return out


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = ["model", "dataset", "method", "display_name", "ablation_meaning"] + METRICS
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "Two-Model DirEAG Ablation",
        "",
        "This ablation uses the same offline model outputs as the main experiments. It isolates internal components of DirEAG rather than comparing against external baselines.",
        "",
        "| Model | Dataset | Variant | Acc | ECE | Brier | AUROC | PR-P | PR-N |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['display_name']} | {r['accuracy']} | {r['ece']} | "
            f"{r['brier']} | {r['auroc']} | {r['auprc_positive']} | {r['auprc_negative']} |"
        )
    lines.extend(
        [
            "",
            "Variant Meaning",
            "",
            "- `Count-only + Cal.`: uses only repeated answer support, then applies final binary calibration.",
            "- `+ Level Reliability`: adds learned reliability weights for the five confidence-steering levels.",
            "- `+ Confidence Evidence`: additionally uses calibrated self-reported confidence as evidence, but removes final binary calibration to expose the raw posterior scale.",
            "- `Full DirEAG`: the complete method; it combines count evidence, level reliability, calibrated confidence evidence, and final selected-answer calibration.",
            "",
            "Short Reading",
            "",
            "Across both models, the final binary calibration is essential for ECE and Brier score: the uncalibrated confidence-evidence posterior often improves or preserves answer selection, but its raw probability scale is poorly calibrated. Confidence evidence contributes most clearly to accuracy on Qwen and Mistral GSM8K, while level reliability alone is usually close to the count-only variant. This suggests that the strongest empirical role of the full model is not merely adding more parameters, but combining calibrated confidence evidence with a final binary calibration layer.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: list[dict[str, str]], path: Path) -> None:
    body = []
    for r in rows:
        body.append(
            f"{r['model']} & {r['dataset']} & {r['display_name']} & {r['accuracy']} & {r['ece']} & "
            f"{r['brier']} & {r['auroc']} & {r['auprc_positive']} & {r['auprc_negative']}\\\\"
        )
    table_rows = "\n".join(body)
    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}

\title{{DirEAG Internal Ablation}}
\author{{DirEAG Study}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Internal Ablation}}

We conduct an internal ablation of DirEAG on two models and three mathematical reasoning datasets. The ablation uses the same offline model outputs as the main experiments and changes only the aggregation and calibration components. The variants are organized by the components they retain. \emph{{Count-only + Cal.}} uses only candidate answer multiplicity and final binary calibration. \emph{{+ Level Reliability}} additionally learns reliability weights for the five confidence-steering levels. \emph{{+ Confidence Evidence}} further uses calibrated self-reported confidence as evidence, but removes final binary calibration to expose the raw posterior scale. \emph{{Full DirEAG}} combines all components, including final selected-answer calibration.

\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{longtable}}{{lllrrrrrr}}
\caption{{Internal ablation of DirEAG components.}}\label{{tab:direag-ablation}}\\
\toprule
Model & Dataset & Variant & Acc & ECE & Brier & AUROC & PR-P & PR-N\\
\midrule
\endfirsthead
\toprule
Model & Dataset & Variant & Acc & ECE & Brier & AUROC & PR-P & PR-N\\
\midrule
\endhead
\bottomrule
\endfoot
{table_rows}
\end{{longtable}}
\normalsize

\paragraph{{Analysis.}}
The ablation shows that the final binary calibration is important for probability quality. The confidence-evidence variant introduces calibrated verbalized confidence into the Dirichlet posterior, but its raw candidate posterior is often poorly calibrated as a final-answer correctness probability, leading to high ECE and Brier score. Adding final calibration substantially reduces this mismatch. Learned steering-level reliability alone is usually close to the count-only variant, suggesting that level weights are not the main source of improvement. Full DirEAG is most useful when calibrated verbalized confidence improves answer selection while the final calibration layer corrects the probability scale.

\end{{document}}
"""
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows, EXP / "outputs" / "tables" / "two_model_dirichlet_ablation.csv")
    write_markdown(rows, EXP / "docs" / "two_model_dirichlet_ablation.md")
    write_latex(rows, EXP / "docs" / "two_model_dirichlet_ablation.tex")
    print(EXP / "outputs" / "tables" / "two_model_dirichlet_ablation.csv")


if __name__ == "__main__":
    main()
