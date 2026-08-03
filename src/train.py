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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    totals = {"loss": 0.0, "iou": 0.0, "acc25": 0.0, "acc50": 0.0, "count": 0.0}
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
    return {key: totals[key] / count for key in ("loss", "iou", "acc25", "acc50")}


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

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = build_loader(args, "train", True)
    val_loader = build_loader(args, "val", False)
    best_iou = float("-inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, criterion, train_loader, optimizer, device, epoch, args.print_freq)
        val_metrics = run_epoch(model, criterion, val_loader, None, device, epoch, args.print_freq)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(args.checkpoint, model, epoch, val_metrics)
            print(f"saved {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
