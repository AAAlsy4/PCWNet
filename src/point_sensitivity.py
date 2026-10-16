from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.PCWNet import PCWNet, PCWNetConfig
from utils.data_loader import RSDataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageNetTransform:
    """Same normalization as demo.py / train.py, applied to test images."""

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        array = image.astype(np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired-prompt IoU evaluation on the test split."
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--data_name", default="CVOGL_DroneAerial")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", default="saved_models/PCWNet_best.pth")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--topk", type=int, default=7)
    parser.add_argument(
        "--margin", type=float, default=0.2,
        help="Normalized distance kept from each border for both click points.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Limit evaluation to the first N samples (-1 = full split).",
    )
    parser.add_argument("--print_freq", type=int, default=50)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> PCWNet:
    """Rebuild PCWNet from the checkpoint config and load its weights."""
    path = Path(args.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    saved = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    valid = {field.name for field in fields(PCWNetConfig)}
    values = {key: value for key, value in saved.items() if key in valid}
    values["topk"] = args.topk
    values["query_patch_scales"] = (
        ((0.25, 0.25), (0.50, 0.30), (0.75, 0.40))
        if args.data_name == "CVOGL_SVI"
        else ((0.25, 0.25),)
    )
    model = PCWNet(PCWNetConfig(**values)).to(device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_prompt_map(
    point_xy: Tuple[float, float],
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Build one Gaussian prompt map on the H_M x W_M prompt grid.

    ``point_xy`` is normalized (x, y) in [0, 1]. This mirrors the map used in
    Eq. (1) of the paper: sigma = 0.1 * sqrt(H_M^2 + W_M^2).
    """
    x, y = point_xy
    ys = torch.arange(grid_h, device=device, dtype=torch.float32)
    xs = torch.arange(grid_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    center_y = y * max(grid_h - 1, 1)
    center_x = x * max(grid_w - 1, 1)
    sigma = 0.1 * math.sqrt(grid_h ** 2 + grid_w ** 2)
    return torch.exp(
        -((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2) / (2.0 * sigma ** 2)
    )


def iou_xyxy(first, second) -> float:
    """IoU between two XYXY boxes in the same coordinate system."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(right - left, 0.0) * max(bottom - top, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-12))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.margin < 0.5:
        raise ValueError("--margin must lie in [0.0, 0.5)")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"device={device}")

    dataset = RSDataset(
        args.data_root,
        args.data_name,
        args.split,
        transform=ImageNetTransform(),
        augment=False,
    )
    model = load_model(args, device)

    total = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))
    if total <= 0:
        raise RuntimeError("No samples to evaluate")

    margin = float(args.margin)
    top_left_xy = (margin, margin)
    bottom_right_xy = (1.0 - margin, 1.0 - margin)

    ious = []
    for index in range(total):
        query, reference, prompt, _, _, _ = dataset[index]
        grid_h, grid_w = prompt.shape[-2], prompt.shape[-1]

        prompt_tl = build_prompt_map(top_left_xy, grid_h, grid_w, device)
        prompt_br = build_prompt_map(bottom_right_xy, grid_h, grid_w, device)
        prompt_batch = torch.stack((prompt_tl, prompt_br), dim=0)  # [2, H_M, W_M]

        query_batch = query[None].to(device).repeat(2, 1, 1, 1)
        reference_batch = reference[None].to(device).repeat(2, 1, 1, 1)

        with torch.inference_mode():
            outputs = model(query_batch, reference_batch, prompt_batch)

        boxes = outputs["boxes"].detach().cpu().numpy()  # [2, 4], normalized XYXY
        pair_iou = iou_xyxy(boxes[0], boxes[1])
        ious.append(pair_iou)

        if (index + 1) % max(args.print_freq, 1) == 0:
            print(
                f"[{index + 1}/{total}] running mean pair IoU = "
                f"{float(np.mean(ious)):.4f}",
                flush=True,
            )

    ious = np.asarray(ious, dtype=np.float64)
    print()
    print(f"data_name       : {args.data_name}")
    print(f"split           : {args.split}")
    print(f"margin          : {margin}")
    print(f"num samples     : {len(ious)}")
    print(f"mean pair IoU   : {ious.mean():.4f}")
    print(f"median pair IoU : {float(np.median(ious)):.4f}")
    print(f"std pair IoU    : {ious.std():.4f}")
    print(f"min / max       : {ious.min():.4f} / {ious.max():.4f}")
    print(f"frac IoU > 0.5  : {float((ious > 0.5).mean()):.4f}")


if __name__ == "__main__":
    main()