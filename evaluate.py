import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import make_model
from data import SPECS, build_dataset


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images)
        correct += int((logits.argmax(1) == labels).sum())
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        total += len(labels)
    if total == 0:
        raise ValueError("Evaluation split is empty")
    return 100 * correct / total, loss_sum / total, total


def checkpoint_config(checkpoint):
    config = checkpoint["config"]
    if config.get("checkpoint_format") != "supplement_final_v1":
        raise ValueError("Unsupported checkpoint format")
    if checkpoint["epoch"] != config["epochs"] - 1:
        raise ValueError("Run is incomplete")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the official dataset files.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    checkpoint_path = args.run_dir / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        config = checkpoint_config(checkpoint)
    except ValueError as error:
        parser.error(str(error))
    dataset_name = config["dataset"]
    model_name = config["model"]
    spec = SPECS[dataset_name]
    if model_name in ("r18", "r50"):
        model = make_model(
            model_name,
            spec["classes"],
            dataset=dataset_name if "fc.weight" in checkpoint["model"] else None,
        )
    else:
        from extra_architectures import make_extra_model

        model = make_extra_model(model_name, spec["classes"], spec["size"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device)

    evaluation_data = build_dataset(
        dataset_name, args.data_root, train=False, augment=False
    )
    loader = DataLoader(
        evaluation_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=args.device == "cuda",
    )
    accuracy, loss, samples = evaluate_model(model, loader, args.device)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_epoch": checkpoint["epoch"],
                "dataset": dataset_name,
                "split": "validation" if dataset_name == "tiny-imagenet" else "test",
                "accuracy": accuracy,
                "loss": loss,
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
