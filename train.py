import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from async_logger import AsyncStepLogger
from common import make_model, seed_all, sync
from controller import Controller
from data import SPECS, build_dataset, identity_hash
from gpu_data import CIFARGPUBatches

ROOT = Path(__file__).resolve().parent


DATASET_DEFAULTS = {
    "cifar10": {
        "epochs": 200,
        "model": "r18",
        "batch_size": 128,
        "label_smoothing": 0.1,
        "seed": 42,
        "weight_decay": 5e-4,
        "nesterov": True,
    },
    "cifar100": {
        "epochs": 200,
        "model": "r18",
        "batch_size": 128,
        "label_smoothing": 0.1,
        "seed": 42,
        "weight_decay": 5e-4,
        "nesterov": True,
    },
    "tiny-imagenet": {
        "epochs": 90,
        "model": "r50",
        "batch_size": 128,
        "label_smoothing": 0.1,
        "seed": 42,
        "weight_decay": 1e-5,
        "nesterov": False,
    },
}


def quantiles(x):
    return [
        float(v)
        for v in torch.quantile(
            x.float(), torch.tensor([0.05, 0.5, 0.95], device=x.device)
        )
    ]


def training_objective(losses, importance_weights, ratio):
    corrected_loss = (importance_weights * losses).mean()
    return corrected_loss, corrected_loss / ratio


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def apply_dataset_defaults(args):
    for name, value in DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "tiny-imagenet"], default="cifar10"
    )
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int)
    p.add_argument("--stop-after-epochs", type=int)
    p.add_argument(
        "--resume", type=Path, help="Epoch checkpoint; --output must be a NEW directory"
    )
    p.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model/controller checkpoints; disable for speed measurements",
    )
    p.add_argument(
        "--model",
        choices=["r18", "r50", "vit_s", "convnext_tiny", "efficientnet_b0"],
    )
    p.add_argument(
        "--training-recipe",
        choices=["legacy", "cross_arch_v2"],
        default="legacy",
        help="Use architecture-specific optimizer/schedule for the cross-architecture rerun.",
    )
    p.add_argument(
        "--lr-override",
        type=float,
        default=None,
        help="Optional recipe LR override, recorded in config.json (for controlled pilots).",
    )
    p.add_argument("--batch-size", type=int)
    p.add_argument(
        "--index-batch-size",
        type=int,
        default=256,
        help="Deprecated compatibility flag; online-cache reindexing performs no encoder forward.",
    )
    p.add_argument("--search-batch-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--sender", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--receiver", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--is-clip", type=float, default=10.0)
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--same-label",
        action="store_true",
        help="Ablation: default follows pseudocode, propagating all edges",
    )
    p.add_argument("--correction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the official dataset files.",
    )
    p.add_argument("--output", type=Path, required=True)
    a = apply_dataset_defaults(p.parse_args())
    if not 0 < a.ratio <= 1 or a.epochs < 1 or a.batch_size < 2 or a.top_k < 1:
        p.error("Invalid budget, epochs, batch size, or top-k")
    if (
        not 0 <= a.beta < 1
        or not 0 <= a.rho <= 1
        or a.lam < 0
        or a.alpha < 0
        or a.is_clip <= 0
        or a.weight_decay < 0
    ):
        p.error("Invalid controller parameters")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    print(f"Loading {a.dataset} training images into host RAM...", flush=True)
    data = CIFARGPUBatches(a.dataset, a.data_root, a.device, a.seed)
    print(f"Loaded {len(data)} training images at {data.size}x{data.size}.", flush=True)
    train_data = build_dataset(a.dataset, a.data_root, train=True, augment=False)
    if not a.top_k < len(data):
        raise ValueError("top-k must be smaller than train set")
    if a.model in ("r18", "r50"):
        model = make_model(a.model, SPECS[a.dataset]["classes"], dataset=a.dataset)
    else:
        from extra_architectures import make_extra_model

        model = make_extra_model(
            a.model, SPECS[a.dataset]["classes"], SPECS[a.dataset]["size"]
        )
    model = model.to(a.device)
    budget = max(2, int(len(data) * a.ratio))
    spe = math.ceil(budget / a.batch_size)
    warmup_spe = math.ceil(len(data) / a.batch_size)
    total_steps = warmup_spe + spe * (a.epochs - 1)
    if a.training_recipe == "cross_arch_v2":
        recipes = {
            "vit_s": {
                "optimizer": "AdamW",
                "lr": 3e-3,
                "weight_decay": 5e-2,
                "warmup_epochs": 10,
                "schedule": "cosine",
            },
            "convnext_tiny": {
                "optimizer": "AdamW",
                "lr": 2e-3,
                "weight_decay": 5e-2,
                "warmup_epochs": 0,
                "schedule": "cosine",
            },
            "efficientnet_b0": {
                "optimizer": "RMSprop",
                "lr": 1e-3,
                "weight_decay": 1e-5,
                "alpha": 0.9,
                "momentum": 0.9,
                "eps": 1e-3,
                "warmup_epochs": 5,
                "schedule": "efficientnet_exp",
            },
        }
        if a.model not in recipes:
            raise ValueError(f"cross_arch_v2 is not defined for model {a.model}")
        train_recipe = dict(recipes[a.model])
        if a.lr_override is not None:
            if a.lr_override <= 0:
                raise ValueError("--lr-override must be positive")
            train_recipe["lr"] = a.lr_override
        if train_recipe["optimizer"] == "AdamW":
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=train_recipe["lr"],
                weight_decay=train_recipe["weight_decay"],
            )
        else:
            opt = torch.optim.RMSprop(
                model.parameters(),
                lr=train_recipe["lr"],
                weight_decay=train_recipe["weight_decay"],
                alpha=train_recipe["alpha"],
                momentum=train_recipe["momentum"],
                eps=train_recipe["eps"],
            )

        def epoch_progress(step):
            if step <= warmup_spe:
                return step / max(warmup_spe, 1)
            return 1.0 + (step - warmup_spe) / max(spe, 1)

        min_lr_ratio = 1e-3

        def lr_multiplier(step):
            ep = epoch_progress(step)
            warmup_epochs = train_recipe["warmup_epochs"]
            if warmup_epochs and ep < warmup_epochs:
                return max(0.05, ep / warmup_epochs)
            decay_ep = max(0.0, ep - warmup_epochs)
            decay_duration = max(1.0, a.epochs - warmup_epochs)
            if train_recipe["schedule"] == "cosine":
                progress = min(1.0, decay_ep / decay_duration)
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            return 0.97 ** (decay_ep / 2.4)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_multiplier)
    else:
        train_recipe = {
            "optimizer": "SGD",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": a.weight_decay,
            "nesterov": a.nesterov,
            "warmup_epochs": 0,
            "schedule": "OneCycleLR",
        }
        opt = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=a.weight_decay,
            nesterov=a.nesterov,
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=0.1,
            total_steps=total_steps,
            div_factor=25,
            final_div_factor=10000,
            pct_start=0.3,
        )
    ctrl = Controller(
        data.targets.to(a.device),
        a.alpha,
        a.rho,
        a.lam,
        a.beta,
        a.is_clip,
        a.label_smoothing,
        a.seed,
        a.same_label,
        a.correction,
        top_k=a.top_k,
        search_batch_size=a.search_batch_size,
        sender=a.sender,
        receiver=a.receiver,
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(
        budget=budget,
        steps_per_budget_epoch=spe,
        steps_in_full_warmup=warmup_spe,
        scheduler_total_steps=total_steps,
        optimizer_config=train_recipe,
        dataset_identity_sha256=identity_hash(train_data),
        train_pool_samples=len(data),
        checkpoint_format="supplement_final_v1",
    )
    exposures = np.zeros(len(data), dtype=np.int64)
    rows = []
    step = seen = reindex_forwards = 0
    start_epoch = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        if ck["config"].get("checkpoint_format") != config["checkpoint_format"]:
            raise ValueError("Resume checkpoint format mismatch")
        for key in (
            "sender",
            "receiver",
            "dataset",
            "ratio",
            "epochs",
            "model",
            "batch_size",
            "top_k",
            "alpha",
            "rho",
            "lam",
            "beta",
            "is_clip",
            "label_smoothing",
            "same_label",
            "correction",
            "seed",
            "training_recipe",
            "dataset_identity_sha256",
            "scheduler_total_steps",
        ):
            if (
                ck["config"].get(key, True if key in ("sender", "receiver") else None)
                != config[key]
            ):
                raise ValueError(f"Resume configuration mismatch: {key}")
        start_epoch = ck["epoch"] + 1
        prior = a.resume.parent
        rows = [
            json.loads(line)
            for line in (prior / "metrics.jsonl").read_text().splitlines()
        ]
        if len(rows) != start_epoch or rows[-1]["epoch"] != ck["epoch"]:
            raise ValueError(
                "Checkpoint/epoch log mismatch; recover aligned exposures before resuming"
            )
        exposures = np.load(prior / "exposures.npy")
        seen = rows[-1]["cumulative_training_samples"]
        if int(exposures.sum()) != seen:
            raise ValueError("Checkpoint/exposure count mismatch")
        step = ck["global_step"]
        if step != rows[-1]["optimizer_steps"]:
            raise ValueError("Checkpoint/step mismatch")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        ctrl.load_state_dict(ck["controller"])
        torch.set_rng_state(ck["torch_rng"])
        if a.device == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        data.generator.set_state(ck["augmentation_rng"])
        for row in rows:
            append(a.output / "metrics.jsonl", row)
        config["resume_parent"] = str(prior.resolve())
        config["resume_epoch"] = start_epoch
        config["step_logs_scope"] = (
            "This segment only; earlier step logs retained in resume_parent"
        )
        del ck
        print(
            f"Resumed checkpoint epoch {start_epoch}; next zero-based epoch {start_epoch}, step {step}",
            flush=True,
        )
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger = AsyncStepLogger(a.output)
    try:
        for epoch in range(start_epoch, min(a.epochs, a.stop_after_epochs or a.epochs)):
            ctrl.new_epoch()
            model.train()
            sync(a.device)
            tick = time.perf_counter()
            correct = sample_count = 0
            loss_sum = corrected_sum = objective_sum = clip_sum = 0.0
            epoch_budget = len(data) if epoch == 0 else budget
            epoch_spe = warmup_spe if epoch == 0 else spe
            loss_ratio = 1.0 if epoch == 0 else a.ratio
            remaining = epoch_budget
            for batch in range(epoch_spe):
                sync(a.device)
                bt = time.perf_counter()
                ids, iw, sampling = ctrl.sample(min(a.batch_size, remaining), step, spe)
                x, y, _ = data.batch(ids)
                logits, features = model(x)
                losses = F.cross_entropy(
                    logits, y, label_smoothing=a.label_smoothing, reduction="none"
                )
                corrected_loss, objective = training_objective(losses, iw, loss_ratio)
                if not torch.isfinite(objective):
                    raise FloatingPointError("Nonfinite objective")
                opt.zero_grad(set_to_none=True)
                objective.backward()
                opt.step()
                sched.step()
                ctrl.cache_embeddings(ids, features, step)
                details = ctrl.observe(ids, logits, y, step)
                sync(a.device)
                elapsed = time.perf_counter() - bt
                if not torch.isfinite(ctrl.w).all():
                    raise FloatingPointError("Nonfinite weights")
                n = len(ids)
                remaining -= n
                sample_count += n
                seen += n
                step += 1
                exposures[ids.cpu().numpy()] += 1
                correct += int((logits.argmax(1) == y).sum())
                loss_sum += float(losses.detach().sum())
                objective_sum += float(objective.detach()) * n
                clip_sum += sampling["is_clip_rate"] * n
                corrected_sum += float(corrected_loss.detach()) * n
                summary = {
                    "epoch": epoch,
                    "batch": batch,
                    "global_step": step,
                    "batch_size": n,
                    "search_candidates": details["search_candidates"],
                    "cache_age_max_steps": details["cache_age_max"],
                    "loss_mean": float(losses.detach().mean()),
                    "objective": float(objective.detach()),
                    "corrected_loss": float(corrected_loss.detach()),
                    "loss_multiplier": 1 / loss_ratio,
                    "batch_accuracy": 100
                    * float((logits.argmax(1) == y).float().mean()),
                    "G_bar": details["G_bar"],
                    "g_hat_p5_p50_p95": quantiles(details["g_hat"]),
                    "weight_p5_p50_p95": quantiles(ctrl.w),
                    "weight_min": float(ctrl.w.min()),
                    "weight_max": float(ctrl.w.max()),
                    "is_clip_rate": sampling["is_clip_rate"],
                    "is_mean": float(iw.mean()),
                    "candidate_count_before_batch": sampling["n_candidates"],
                    "stale_max": sampling["stale_max"],
                    "n_touched": details["n_touched"],
                    "info_ratio": details["info_ratio"],
                    "step_seconds": elapsed,
                    "lr": opt.param_groups[0]["lr"],
                    "available": int(ctrl.available.sum()),
                }
                samples = dict(
                    epoch=epoch,
                    batch=batch,
                    global_step=step,
                    sample_ids=ids.cpu().tolist(),
                    targets=y.cpu().tolist(),
                    losses=losses.detach().cpu().tolist(),
                    sampling_probabilities=sampling["probabilities"].cpu().tolist(),
                    is_weights=iw.cpu().tolist(),
                    raw_is=sampling["raw_is"].cpu().tolist(),
                    staleness=sampling["stale"].cpu().tolist(),
                    candidate_count_before_batch=sampling["n_candidates"],
                    **{
                        k: v.cpu().tolist()
                        for k, v in details.items()
                        if isinstance(v, torch.Tensor)
                    },
                )
                logger.write(summary, samples)
            assert (
                sample_count == epoch_budget
                and int((~ctrl.available).sum()) == epoch_budget
            )
            logger.flush()
            sync(a.device)
            train_seconds = time.perf_counter() - tick
            row = {
                "epoch": epoch,
                "selected_samples": sample_count,
                "cumulative_training_samples": seen,
                "optimizer_steps": step,
                "train_accuracy": 100 * correct / sample_count,
                "train_loss": loss_sum / sample_count,
                "objective": objective_sum / sample_count,
                "corrected_loss": corrected_sum / sample_count,
                "loss_multiplier": 1 / loss_ratio,
                "train_seconds": train_seconds,
                "reindex_seconds": 0.0,
                "cumulative_embedding_forwards": reindex_forwards,
                "is_clip_rate": clip_sum / sample_count,
                "weight_p5_p50_p95": quantiles(ctrl.w),
                "unique_seen": int((exposures > 0).sum()),
            }
            rows.append(row)
            append(a.output / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            np.save(a.output / "exposures.npy", exposures)
            if a.save_checkpoints:
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "controller": ctrl.state_dict(),
                    "config": config,
                    "global_step": step,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if a.device == "cuda"
                    else None,
                    "augmentation_rng": data.generator.get_state(),
                }
                torch.save(ckpt, a.output / "last.tmp")
                (a.output / "last.tmp").replace(a.output / "last.pt")
    finally:
        logger.close()
    result = {
        "complete": len(rows) == a.epochs,
        "checkpoints_saved": a.save_checkpoints,
        "completed_epochs": len(rows),
        "final_training_accuracy": rows[-1]["train_accuracy"],
        "final_training_loss": rows[-1]["train_loss"],
        "training_sample_forwards": seen,
        "training_sample_backwards": seen,
        "extra_embedding_forwards": reindex_forwards,
        "total_wall_seconds": None if a.resume else time.perf_counter() - start,
        "segment_wall_seconds": time.perf_counter() - start,
        "resume_parent": str(a.resume.parent.resolve()) if a.resume else None,
        "wall_time_note": "Prior segment total wall unavailable; downtime excluded, do not treat segment wall as full run"
        if a.resume
        else None,
        "total_train_seconds": sum(r["train_seconds"] for r in rows),
        "total_reindex_seconds": sum(r["reindex_seconds"] for r in rows),
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
