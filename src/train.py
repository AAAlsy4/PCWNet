from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Dict, Iterable, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.loss import Criterion, aligned_iou, normalize_xyxy_boxes
from model.PCWNet import PCWNetConfig, PCWNet
from utils.data_loader import RSDataset


WARMUP_EPOCHS = 2


def parse_args() -> argparse.Namespace:
    """Parse command-line options for training or evaluation."""
    parser = argparse.ArgumentParser(description="Train PCWNet")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--data_name", default="CVOGL_DroneAerial")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=7)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--checkpoint", default="saved_models/PCWNet_best.pth")
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--print_freq", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training schedules before allocating model resources."""
    if args.eval:
        return
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.finetune_epochs < 0:
        raise ValueError("--finetune_epochs cannot be negative")
    if args.lr <= 0 or args.backbone_lr <= 0 or args.min_lr < 0:
        raise ValueError("learning rates must be positive (--min_lr may be zero)")
    if args.finetune_lr is not None and args.finetune_lr <= 0:
        raise ValueError("--finetune_lr must be positive")


def build_model(args: argparse.Namespace) -> PCWNet:
    """Construct PCWNet with candidate reranking enabled for all datasets."""
    return PCWNet(PCWNetConfig(
        topk=args.topk,
        query_patch_scales=((0.25, 0.25), (0.50, 0.30), (0.75, 0.40)) if args.data_name == "CVOGL_SVI" else ((0.25, 0.25),),
    ))


def build_criterion() -> Criterion:
    """Construct the training criterion with its default loss weights."""
    return Criterion()


def build_optimizer(
    args: argparse.Namespace,
    model: PCWNet,
    lr: float | None = None,
    backbone_lr: float | None = None,
) -> torch.optim.Optimizer:
    """Create AdamW parameter groups with a separate fine-backbone learning rate."""
    head_lr = args.lr if lr is None else lr
    fine_backbone_lr = args.backbone_lr if backbone_lr is None else backbone_lr
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("query_fine_encoder.", "reference_fine_encoder.")):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = [{"params": head_parameters, "lr": head_lr}]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": fine_backbone_lr}
        )
    return torch.optim.AdamW(
        parameter_groups,
        lr=head_lr,
    )


def build_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    epochs: int | None = None,
    lr: float | None = None,
    warmup_epochs: int = WARMUP_EPOCHS,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear-warmup and cosine-decay learning-rate scheduler."""
    total_epochs = args.epochs if epochs is None else epochs
    base_lr = args.lr if lr is None else lr
    warmup = warmup_epochs
    warmup = min(max(warmup, 0), max(total_epochs - 1, 0))
    min_factor = min(max(args.min_lr / max(base_lr, 1e-12), 0.0), 1.0)

    def schedule(epoch: int) -> float:
        """Return the multiplicative learning-rate factor for one epoch."""
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        decay_epochs = max(total_epochs - warmup - 1, 1)
        progress = min(max((epoch - warmup) / decay_epochs, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_factor + (1.0 - min_factor) * float(cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_dataset(args: argparse.Namespace, split: str, augment: bool) -> RSDataset:
    """Build one dataset split with the requested augmentation mode."""
    return RSDataset(
        data_root=args.data_root,
        data_name=args.data_name,
        split_name=split,
        transform=ImageNetTransform(),
        augment=augment,
    )


def build_data_loader(
    args: argparse.Namespace, dataset: Dataset, shuffle: bool
) -> DataLoader:
    """Wrap a dataset in the common loader configuration."""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_loader(args: argparse.Namespace, split: str, augment: bool) -> DataLoader:
    """Build a dataset loader for the requested split and augmentation mode."""
    return build_data_loader(
        args, build_dataset(args, split, augment), shuffle=augment
    )


def build_train_val_loader(args: argparse.Namespace) -> DataLoader:
    """Build one shuffled training loader containing train and val samples."""
    dataset = ConcatDataset(
        [
            build_dataset(args, "train", augment=True),
            build_dataset(args, "val", augment=True),
        ]
    )
    return build_data_loader(args, dataset, shuffle=True)


class ImageNetTransform:
    """Convert RGB uint8 images to ImageNet-normalized CHW tensors."""

    def __init__(self) -> None:
        """Store ImageNet channel statistics as broadcastable tensors."""
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Normalize an ``[H, W, 3]`` RGB array into ``[3, H, W]``."""
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.float().div(255.0)
        return (tensor - self.mean) / self.std


def move_batch(batch: Tuple[object, ...], device: torch.device):
    """Move tensor fields of one dataset batch to the selected device."""
    query, reference, prompt, boxes, indices, classes = batch
    return (
        query.to(device, non_blocking=True),
        reference.to(device, non_blocking=True),
        prompt.to(device, non_blocking=True),
        boxes.to(device, non_blocking=True),
        indices,
        classes,
    )


def run_epoch(
    model: PCWNet,
    criterion: Criterion,
    loader: Iterable[Tuple[object, ...]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    print_freq: int,
) -> Dict[str, float]:
    """Run one training or evaluation epoch and aggregate localization metrics."""
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "iou": 0.0,
        "acc25": 0.0,
        "acc50": 0.0,
        "oracle_iou": 0.0,
        "oracle_acc25": 0.0,
        "oracle_acc50": 0.0,
        "selection_acc": 0.0,
        "selection_regret": 0.0,
        "count": 0.0,
    }
    start = time.time()
    for batch_index, raw_batch in enumerate(loader):
        query, reference, prompt, boxes, _, _ = move_batch(raw_batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(query, reference, prompt)
            losses = criterion(outputs, boxes, image_size=reference.shape[-2:])
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        target = normalize_xyxy_boxes(boxes, reference.shape[-2:])
        ious = aligned_iou(outputs["boxes"].detach(), target)
        n = query.shape[0]
        totals["loss"] += float(losses["loss"].detach()) * n
        totals["iou"] += float(ious.sum())
        totals["acc25"] += float((ious >= 0.25).sum())
        totals["acc50"] += float((ious >= 0.50).sum())
        candidate_boxes = outputs.get("candidate_boxes")
        if candidate_boxes is not None:
            candidate_targets = target[:, None].expand_as(candidate_boxes)
            candidate_ious = aligned_iou(candidate_boxes.detach(), candidate_targets)
            oracle_ious, oracle_indices = candidate_ious.max(dim=1)
            totals["oracle_iou"] += float(oracle_ious.sum())
            totals["oracle_acc25"] += float((oracle_ious >= 0.25).sum())
            totals["oracle_acc50"] += float((oracle_ious >= 0.50).sum())
            totals["selection_acc"] += float(
                (outputs["selected_indices"].detach() == oracle_indices).sum()
            )
            totals["selection_regret"] += float((oracle_ious - ious).sum())
        totals["count"] += n
        if batch_index % max(print_freq, 1) == 0:
            mode = "train" if training else "eval"
            print(
                f"[{mode}] epoch={epoch + 1} batch={batch_index}/{len(loader)} "
                f"loss={float(losses['loss'].detach()):.4f} "
                f"iou={float(ious.mean()):.4f} "
                f"acc50={float((ious >= 0.50).float().mean()):.4f} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    count = max(totals["count"], 1.0)
    return {
        key: totals[key] / count
        for key in (
            "loss",
            "iou",
            "acc25",
            "acc50",
            "oracle_iou",
            "oracle_acc25",
            "oracle_acc50",
            "selection_acc",
            "selection_regret",
        )
    }


def checkpoint_score(metrics: Dict[str, float]) -> float:
    """Prioritize the stricter threshold while retaining acc25 as a guardrail."""
    return 0.25 * metrics["acc25"] + 0.75 * metrics["acc50"]


def save_checkpoint(
    path: str,
    model: PCWNet,
    epoch: int,
    metrics: Dict[str, float],
    training_state: Dict[str, object] | None = None,
) -> None:
    """Persist model weights, epoch metadata, configuration, and metrics."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "config": asdict(model.config),
    }
    if training_state is not None:
        checkpoint["training_state"] = training_state
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: torch.nn.Module) -> Dict[str, object]:
    """Load a checkpoint into ``model`` and return its stored metadata."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    return checkpoint


def finetune_on_train_val(
    args: argparse.Namespace,
    model: PCWNet,
    criterion: Criterion,
    device: torch.device,
) -> None:
    """Fine-tune the best validation model on the combined train+val data."""
    if args.finetune_epochs == 0:
        print("train+val fine-tuning disabled", flush=True)
        return

    source = load_checkpoint(args.checkpoint, model)
    source_epoch = int(source.get("epoch", -1))
    source_metrics = source.get("metrics", {})
    finetune_lr = args.finetune_lr
    if finetune_lr is None:
        finetune_lr = args.lr * 0.1
    finetune_backbone_lr = args.backbone_lr * (finetune_lr / args.lr)

    optimizer = build_optimizer(
        args,
        model,
        lr=finetune_lr,
        backbone_lr=finetune_backbone_lr,
    )
    scheduler = build_scheduler(
        args,
        optimizer,
        epochs=args.finetune_epochs,
        lr=finetune_lr,
        warmup_epochs=0,
    )
    loader = build_train_val_loader(args)
    print(f"loaded best checkpoint {args.checkpoint}", flush=True)

    for finetune_epoch in range(args.finetune_epochs):
        metrics = run_epoch(
            model,
            criterion,
            loader,
            optimizer,
            device,
            finetune_epoch,
            args.print_freq,
        )
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        print(
            f"finetune_epoch={finetune_epoch + 1}\n"
            f"train+val={metrics}\nlr={current_lr:.6g}",
            flush=True,
        )
        scheduler.step()
        base, ext = os.path.splitext(args.checkpoint)
        fine_checkpoint = f"{base}_fine_{finetune_epoch + 1}{ext}"
        save_checkpoint(
            fine_checkpoint,
            model,
            args.epochs + finetune_epoch,
            metrics,
            training_state={
                "stage": "train_val_finetune",
                "source_checkpoint": args.checkpoint,
                "source_epoch": source_epoch,
                "source_metrics": source_metrics,
                "finetune_epoch": finetune_epoch,
                "finetune_epochs": args.finetune_epochs,
                "learning_rate": current_lr,
            },
        )
        print(f"saved {fine_checkpoint}", flush=True)


def main() -> None:
    """Run the configured training loop or one evaluation pass."""
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"device={device}")
    model = build_model(args).to(device)
    criterion = build_criterion().to(device)
    if args.eval:
        if not args.resume:
            raise ValueError("--eval requires --resume")
        checkpoint = load_checkpoint(args.resume, model)
        print("loaded checkpoint")
        metrics = run_epoch(
            model, criterion, build_loader(args, args.split, False), None, device, 0,
            args.print_freq,
        )
        print(json.dumps(metrics, indent=2))
        return

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    train_loader = build_loader(args, "train", True)
    val_loader = build_loader(args, "val", False)
    best_score = float("-inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, criterion, train_loader, optimizer, device, epoch, args.print_freq,
        )
        val_metrics = run_epoch(
            model, criterion, val_loader, None, device, epoch, args.print_freq,
        )
        score = checkpoint_score(val_metrics)
        val_metrics["selection_score"] = score
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        print(
            f"epoch={epoch + 1}\ntrain={train_metrics}\nval={val_metrics}\n"
            f"lr={current_lr:.6g}",
            flush=True,
        )
        scheduler.step()
        if best_epoch < 0 or (
            np.isfinite(score)
            and (not np.isfinite(best_score) or score > best_score)
        ):
            best_score = score
            best_epoch = epoch
            save_checkpoint(args.checkpoint, model, epoch, val_metrics)
            print(f"saved {args.checkpoint}", flush=True)

    print(
        f"best validation checkpoint: epoch={best_epoch + 1} "
        f"score={best_score:.6f}",
        flush=True,
    )
    finetune_on_train_val(args, model, criterion, device)


if __name__ == "__main__":
    main()

