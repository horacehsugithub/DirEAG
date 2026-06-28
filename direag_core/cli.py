from __future__ import annotations

import argparse

from .data import prepare_all_datasets
from .dirichlet import run_analysis
from .inference import run_inference
from .report import compile_report, write_report
from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the second Dirichlet confidence experiment.")
    parser.add_argument("command", choices=["prepare-data", "infer", "analyze", "report", "all"])
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command in {"prepare-data", "all"}:
        paths = prepare_all_datasets(config)
        for path in paths:
            print(f"prepared {path}")
    if args.command in {"infer", "all"}:
        paths = run_inference(config)
        for path in paths:
            print(f"wrote {path}")
    if args.command in {"analyze", "all"}:
        paths = run_analysis(config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    if args.command == "report":
        write_report(config)
        compile_report(config)
    elif args.command == "all":
        compile_report(config)


if __name__ == "__main__":
    main()
