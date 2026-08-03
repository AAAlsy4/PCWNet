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
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.loss import Criterion, LossConfig, aligned_iou, normalize_xyxy_boxes
from model.PCWNet import PCWNetConfig, PCWNet
from utils.data_loader import RSDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCWNet")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--data_name", default="CVOGL_DroneAerial")
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--no_pretrained_backbones", dest="pretrained_backbones", action="store_false")
    parser.add_argument("--train_coarse_backbone", dest="freeze_coarse", action="store_false")
    parser.set_defaults(pretrained_backbones=True, freeze_coarse=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--checkpoint", default="saved_models/PCWNet_best.pth")
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--print_freq", type=int, default=100)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> PCWNet:
    return PCWNet(PCWNetConfig(
        pretrained_backbones=args.pretrained_backbones,
        freeze_coarse=args.freeze_coarse and args.pretrained_backbones,
    ))


def build_optimizer(args: argparse.Namespace, model: PCWNet) -> torch.optim.Optimizer:
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("query_fine_encoder.", "reference_fine_encoder.")):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    return torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = min(max(args.warmup_epochs, 0), max(args.epochs - 1, 0))
    min_factor = min(max(args.min_lr / max(args.lr, 1e-12), 0.0), 1.0)

    def schedule(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        decay_epochs = max(args.epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_factor + (1.0 - min_factor) * float(cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_loader(args: argparse.Namespace, split: str, augment: bool) -> DataLoader:

    dataset = RSDataset(
        data_root=args.data_root,
        data_name=args.data_name,
        split_name=split,
        img_size=args.img_size,
        transform=ImageNetTransform(),
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=augment,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class ImageNetTransform:
    def __init__(self) -> None:
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.float().div(255.0)
        return (tensor - self.mean) / self.std


def move_batch(batch: Tuple[object, ...], device: torch.device):
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
        totals["count"] += n
        if batch_index % max(print_freq, 1) == 0:
            mode = "train" if training else "eval"
            print(
                f"[{mode}] epoch={epoch} batch={batch_index}/{len(loader)} "
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
        )
    }


def checkpoint_score(metrics: Dict[str, float]) -> float:
    """Prioritize the stricter threshold while retaining acc25 as a guardrail."""
    return 0.4 * metrics["acc25"] + 0.6 * metrics["acc50"]


def save_checkpoint(path: str, model: PCWNet, epoch: int, metrics: Dict[str, float]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                "metrics": metrics, "config": asdict(model.config)}, path)


def load_checkpoint(path: str, model: torch.nn.Module) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"device={device}")
    model = build_model(args).to(device)
    criterion = Criterion(LossConfig()).to(device)
    if args.eval:
        if not args.resume:
            raise ValueError("--eval requires --resume")
        checkpoint = load_checkpoint(args.resume, model)
        print(f"loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}")
        metrics = run_epoch(model, criterion, build_loader(args, args.split, False), None, device, 0, args.print_freq)
        print(json.dumps(metrics, indent=2))
        return

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    train_loader = build_loader(args, "train", True)
    val_loader = build_loader(args, "val", False)
    best_score = float("-inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, criterion, train_loader, optimizer, device, epoch, args.print_freq)
        val_metrics = run_epoch(model, criterion, val_loader, None, device, epoch, args.print_freq)
        score = checkpoint_score(val_metrics)
        val_metrics["selection_score"] = score
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        print(
            f"epoch={epoch} train={train_metrics} val={val_metrics} "
            f"lr={current_lr:.6g}",
            flush=True,
        )
        scheduler.step()
        if score > best_score:
            best_score = score
            save_checkpoint(args.checkpoint, model, epoch, val_metrics)
            print(f"saved {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
