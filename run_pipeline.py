"""Prepare data, train, and evaluate the completed final checkpoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "tiny-imagenet", "all"],
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be in (0, 1]")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    datasets = (
        ["cifar10", "cifar100", "tiny-imagenet"]
        if args.dataset == "all"
        else [args.dataset]
    )
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    runs = [
        output_root / f"{dataset}_r{args.ratio:g}_s{args.seed}" for dataset in datasets
    ]
    for run in runs:
        if run.exists() and any(run.iterdir()):
            parser.error(f"Run directory must be empty: {run}")
    for dataset, run in zip(datasets, runs):
        if args.prepare_data:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "prepare_data.py"),
                    "--dataset",
                    dataset,
                    "--data-root",
                    str(data_root),
                ],
                check=True,
            )
        command = [
            sys.executable,
            "-u",
            str(ROOT / "train.py"),
            "--dataset",
            dataset,
            "--ratio",
            str(args.ratio),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--data-root",
            str(data_root),
            "--output",
            str(run),
        ]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        subprocess.run(command, check=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "evaluate.py"),
                "--run-dir",
                str(run),
                "--data-root",
                str(data_root),
                "--device",
                args.device,
                "--batch-size",
                str(args.eval_batch_size),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        (run / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
