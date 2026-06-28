from __future__ import annotations

import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs" / "tables"
FIGS = Path(__file__).resolve().parents[1] / "outputs" / "figures"
DOCS = Path(__file__).resolve().parents[1] / "docs"

MAIN_METHOD = "mle_dirichlet_qcal_platt"

SOURCES = [
    {
        "model": "Qwen2.5-7B-Instruct",
        "dataset_filter": "gsm8k",
        "path": ROOT / "results_gsm8kfull" / "outputs" / "tables" / "second_experiment_predictions.csv",
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "dataset_filter": "svamp",
        "path": ROOT / "svamp_second_experiment" / "outputs" / "tables" / "second_experiment_predictions.csv",
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "dataset_filter": "gsmhard",
        "path": ROOT / "gsmhard" / "outputs" / "tables" / "second_experiment_predictions.csv",
    },
    {
        "model": "Mistral-7B-Instruct-v0.3",
        "dataset_filter": None,
        "path": ROOT
        / "20260524_result"
        / "mistral_dirichlet_experiment"
        / "outputs"
        / "tables"
        / "second_experiment_predictions.csv",
    },
    {
        "model": "Gemma-2-9B-IT",
        "dataset_filter": None,
        "path": ROOT
        / "gemma"
        / "gemma_three_dataset_experiment"
        / "outputs"
        / "tables"
        / "second_experiment_predictions.csv",
    },
]

DATASET_LABEL = {
    "gsm8k": "GSM8K",
    "svamp": "SVAMP",
    "gsmhard": "GSM-Hard",
}


def clip01(x: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(x)))


def ece_equal_width(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = len(y)
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        if i == 0:
            mask = (lo <= p) & (p <= hi)
        else:
            mask = (lo < p) & (p <= hi)
        if mask.any():
            out += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(out)


def adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p)
    out = 0.0
    n = len(order)
    for b in range(bins):
        lo = round(b * n / bins)
        hi = round((b + 1) * n / bins)
        idx = order[lo:hi]
        if len(idx):
            out += len(idx) / n * abs(y[idx].mean() - p[idx].mean())
    return float(out)


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    y = rows["correct"].astype(float).to_numpy()
    p = rows["confidence"].astype(float).map(clip01).to_numpy()
    return {
        "n": int(len(rows)),
        "accuracy": float(y.mean()),
        "mean_confidence": float(p.mean()),
        "confidence_sd": float(np.std(p, ddof=0)),
        "ece": ece_equal_width(y, p),
        "adaptive_ece": adaptive_ece(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "auroc": safe_auroc(y, p),
        "prp": safe_ap(y, p),
        "prn": safe_ap(1.0 - y, 1.0 - p),
    }


def load_direag_predictions() -> pd.DataFrame:
    frames = []
    for src in SOURCES:
        df = pd.read_csv(src["path"])
        df = df[df["method"] == MAIN_METHOD].copy()
        if src["dataset_filter"] is not None:
            df = df[df["dataset"] == src["dataset_filter"]].copy()
        df["model"] = src["model"]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = out["dataset"].map(lambda x: str(x).lower())
    out["dataset_label"] = out["dataset"].map(DATASET_LABEL)
    out["correct"] = out["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    out["confidence"] = out["confidence"].astype(float).map(clip01)
    return out


def build_diagnostic_predictions(direag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset), group in direag.groupby(["model", "dataset"], sort=False):
        base_rate = float(group["correct"].mean())

        d = group.copy()
        d["diagnostic_method"] = "DirEAG"
        d["base_rate"] = base_rate
        rows.append(d)

        b = group.copy()
        b["diagnostic_method"] = "Oracle base-rate"
        b["base_rate"] = base_rate
        b["confidence"] = clip01(base_rate)
        rows.append(b)

    out = pd.concat(rows, ignore_index=True)
    keep = [
        "model",
        "dataset",
        "dataset_label",
        "fold",
        "problem_id",
        "diagnostic_method",
        "pred_answer_norm",
        "gold_answer_raw",
        "correct",
        "confidence",
        "base_rate",
    ]
    return out[keep]


def make_metric_tables(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    for (model, dataset, method), group in preds.groupby(
        ["model", "dataset", "diagnostic_method"], sort=False
    ):
        row = summarize(group)
        row.update(
            {
                "model": model,
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "diagnostic_method": method,
            }
        )
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    cols = [
        "model",
        "dataset",
        "dataset_label",
        "diagnostic_method",
        "n",
        "accuracy",
        "mean_confidence",
        "confidence_sd",
        "ece",
        "adaptive_ece",
        "brier",
        "nll",
        "auroc",
        "prp",
        "prn",
    ]
    metrics = metrics[cols]

    delta_rows = []
    for (model, dataset), group in metrics.groupby(["model", "dataset"], sort=False):
        d = group[group["diagnostic_method"] == "DirEAG"].iloc[0]
        b = group[group["diagnostic_method"] == "Oracle base-rate"].iloc[0]
        row = {
            "model": model,
            "dataset": dataset,
            "dataset_label": DATASET_LABEL[dataset],
            "n": int(d["n"]),
            "accuracy": float(d["accuracy"]),
            "base_rate": float(b["mean_confidence"]),
            "delta_brier": float(d["brier"] - b["brier"]),
            "delta_nll": float(d["nll"] - b["nll"]),
            "delta_auroc": float(d["auroc"] - b["auroc"]),
            "delta_prp": float(d["prp"] - b["prp"]),
            "delta_prn": float(d["prn"] - b["prn"]),
            "direag_ece": float(d["ece"]),
            "base_ece": float(b["ece"]),
            "direag_confidence_sd": float(d["confidence_sd"]),
        }
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    return metrics, deltas


def fmt(x: float, digits: int = 3) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(fmt(float(value), digits))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lines = [
        "Instance-Level Uncertainty Diagnostic",
        "",
        "This diagnostic compares DirEAG with an oracle constant base-rate baseline. "
        "For each model-dataset pair, the baseline keeps the same final-answer distribution "
        "diagnostically by using the empirical correctness rate of DirEAG as a single constant confidence for every instance. "
        "It is not a deployable calibration method; it is a stress test for whether DirEAG contributes instance-level uncertainty.",
        "",
        "Metrics",
        "",
        dataframe_to_markdown(metrics, digits=4),
        "",
        "DirEAG minus Oracle Base-Rate",
        "",
        "Negative deltas are better for Brier and NLL; positive deltas are better for AUROC, PR-P, and PR-N.",
        "",
        dataframe_to_markdown(deltas, digits=4),
        "",
    ]
    (DOCS / "instance_level_uncertainty_diagnostic.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def short_model_name(name: str) -> str:
    return (
        name.replace("Qwen2.5-7B-Instruct", "Qwen")
        .replace("Mistral-7B-Instruct-v0.3", "Mistral")
        .replace("Gemma-2-9B-IT", "Gemma")
    )


def plot_instance_level_diagnostic(deltas: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    model_order = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Gemma-2-9B-IT"]
    dataset_order = ["gsm8k", "svamp", "gsmhard"]
    dataset_labels = [DATASET_LABEL[d] for d in dataset_order]
    model_labels = [short_model_name(m) for m in model_order]
    panels = [
        ("Brier reduction", "brier_reduction", lambda r: -float(r["delta_brier"]), "tab:blue"),
        ("NLL reduction", "nll_reduction", lambda r: -float(r["delta_nll"]), "tab:green"),
        ("AUROC gain", "auroc_gain", lambda r: float(r["delta_auroc"]), "tab:red"),
    ]

    values: dict[str, np.ndarray] = {}
    for _, key, getter, _ in panels:
        mat = np.zeros((len(model_order), len(dataset_order)), dtype=float)
        for i, model in enumerate(model_order):
            for j, dataset in enumerate(dataset_order):
                row = deltas[(deltas["model"] == model) & (deltas["dataset"] == dataset)].iloc[0]
                mat[i, j] = getter(row)
        values[key] = mat

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)
    for ax, (title, key, _, color) in zip(axes, panels):
        mat = values[key]
        vmax = max(0.01, float(np.nanmax(mat)))
        im = ax.imshow(mat, cmap="Blues" if "Brier" in title else ("Greens" if "NLL" in title else "Reds"), vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks(np.arange(len(dataset_labels)))
        ax.set_xticklabels(dataset_labels, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(model_labels)))
        ax.set_yticklabels(model_labels if ax is axes[0] else [""] * len(model_labels), fontsize=9)
        ax.tick_params(length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                text_color = "white" if mat[i, j] > 0.62 * vmax else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9, color=text_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8, length=0)

    fig.suptitle("Instance-Level Gains over a Constant Base Rate", fontsize=11)
    fig.savefig(FIGS / "instance_level_base_rate_diagnostic.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "instance_level_base_rate_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_direct_comparison(metrics: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    model_order = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Gemma-2-9B-IT"]
    dataset_order = ["gsm8k", "svamp", "gsmhard"]
    pairs = [(m, d) for m in model_order for d in dataset_order]
    labels = [f"{short_model_name(m)}\n{DATASET_LABEL[d]}" for m, d in pairs]
    x = np.arange(len(pairs))

    panels = [
        ("Brier score", "brier", "lower"),
        ("AUROC", "auroc", "higher"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    for ax, (title, metric, direction) in zip(axes, panels):
        base_vals = []
        direag_vals = []
        for model, dataset in pairs:
            sub = metrics[(metrics["model"] == model) & (metrics["dataset"] == dataset)]
            base_vals.append(float(sub[sub["diagnostic_method"] == "Oracle base-rate"][metric].iloc[0]))
            direag_vals.append(float(sub[sub["diagnostic_method"] == "DirEAG"][metric].iloc[0]))
        base_vals = np.asarray(base_vals)
        direag_vals = np.asarray(direag_vals)

        for i, (b, d) in enumerate(zip(base_vals, direag_vals)):
            ax.plot([i, i], [b, d], color="darkgray", linewidth=1.4, zorder=1)
            ax.annotate(
                "",
                xy=(i, d),
                xytext=(i, b),
                arrowprops=dict(arrowstyle="-|>", color="darkgray", lw=1.0, shrinkA=3, shrinkB=3),
                zorder=1,
            )
        base_scatter = ax.scatter(x, base_vals, s=46, color="gray", label="Base-rate", zorder=3)
        direag_scatter = ax.scatter(x, direag_vals, s=58, color="tab:red", label="DirEAG", zorder=3)
        ax.set_title(f"{title} ({direction} is better)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "auroc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.8)
    fig.legend(
        handles=[base_scatter, direag_scatter],
        labels=["Base-rate", "DirEAG"],
        loc="center left",
        bbox_to_anchor=(1.005, 0.55),
        ncol=1,
        frameon=False,
        fontsize=9,
    )

    fig.savefig(FIGS / "instance_level_base_rate_direct_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "instance_level_base_rate_direct_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex_section(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    min_brier = -float(deltas["delta_brier"].max())
    max_brier = -float(deltas["delta_brier"].min())
    min_auroc = float(deltas["delta_auroc"].min())
    max_auroc = float(deltas["delta_auroc"].max())
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{times}}
\usepackage{{microtype}}
\begin{{document}}

\section{{Instance-Level Uncertainty Diagnostic}}

A possible concern is that a learned confidence model may only recover a dataset-level base rate rather than provide meaningful instance-level uncertainty. We therefore compare DirEAG with a constant base-rate diagnostic. For each model--dataset pair, the diagnostic baseline assigns every evaluated instance the same confidence, equal to the empirical correctness rate of DirEAG on that pair:
\[
    \hat c_i^{{\mathrm{{base}}}} = \bar y
    = \frac{{1}}{{n}}\sum_{{i=1}}^n \mathbf{{1}}[\hat a_i = y_i] .
\]
This baseline is intentionally strong as a diagnostic: among constant confidence values, it is optimal for squared error and log loss on the evaluated set. It is not intended as a deployable calibration method. Instead, it asks whether DirEAG's instance-varying confidence scores contain information beyond the average success rate. Since the constant baseline has no ranking ability, its AUROC is $0.5$ whenever both correct and incorrect examples are present, and its PR scores reduce to class prevalences.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../outputs/figures/instance_level_base_rate_direct_comparison.pdf}}
\caption{{Direct comparison between DirEAG and the oracle constant base-rate baseline. Each vertical pair corresponds to one model--dataset setting. For Brier score, lower values are better; for AUROC, higher values are better. The constant baseline has no instance-level ranking ability, so its AUROC is $0.5$.}}
\label{{fig:instance-level-diagnostic}}
\end{{figure}}

Figure~\ref{{fig:instance-level-diagnostic}} shows that DirEAG improves over the constant base-rate baseline in Brier score across all nine model--dataset pairs, despite the baseline being allowed to use the empirical average correctness rate. The Brier reduction ranges from {min_brier:.3f} to {max_brier:.3f}. More importantly, DirEAG obtains AUROC gains from {min_auroc:.3f} to {max_auroc:.3f} over the constant baseline, which has no ability to distinguish easier from harder instances. These results indicate that the learned Dirichlet aggregation does not merely memorize the marginal accuracy of a dataset; it uses the pattern of steered answer-confidence observations to assign different uncertainty levels to different problem instances.

The ECE of the oracle base-rate baseline is close to zero by construction, because all examples fall into a single confidence bin whose mean confidence equals the empirical accuracy. We therefore do not interpret ECE in this diagnostic as evidence against instance-level uncertainty. The relevant quantities are proper scoring and ranking metrics, where the constant baseline cannot benefit from item-specific information.

\end{{document}}
"""
    (DOCS / "instance_level_uncertainty_section.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    direag = load_direag_predictions()
    preds = build_diagnostic_predictions(direag)
    metrics, deltas = make_metric_tables(preds)

    preds.to_csv(OUT / "instance_level_diagnostic_predictions.csv", index=False)
    metrics.to_csv(OUT / "instance_level_diagnostic_metrics.csv", index=False)
    deltas.to_csv(OUT / "instance_level_diagnostic_deltas.csv", index=False)
    plot_instance_level_diagnostic(deltas)
    plot_direct_comparison(metrics)
    write_markdown(metrics, deltas)
    write_latex_section(metrics, deltas)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "instance_level_uncertainty_section.tex"], cwd=DOCS, check=False)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "instance_level_uncertainty_section.tex"], cwd=DOCS, check=False)

    print("Wrote:")
    print(OUT / "instance_level_diagnostic_predictions.csv")
    print(OUT / "instance_level_diagnostic_metrics.csv")
    print(OUT / "instance_level_diagnostic_deltas.csv")
    print(FIGS / "instance_level_base_rate_diagnostic.png")
    print(FIGS / "instance_level_base_rate_diagnostic.pdf")
    print(FIGS / "instance_level_base_rate_direct_comparison.png")
    print(FIGS / "instance_level_base_rate_direct_comparison.pdf")
    print(DOCS / "instance_level_uncertainty_diagnostic.md")
    print(DOCS / "instance_level_uncertainty_section.tex")
    print(DOCS / "instance_level_uncertainty_section.pdf")


if __name__ == "__main__":
    main()
