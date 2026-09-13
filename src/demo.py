from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.PCWNet import PCWNet, PCWNetConfig
from utils.data_loader import RSDataset
from utils.utils import states_to_boxes


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CANDIDATE_COLORS = (
    (0, 114, 178),
    (230, 159, 0),
    (204, 121, 167),
    (86, 180, 233),
    (240, 228, 66),
    (128, 0, 128),
    (0, 128, 128),
)
GT_COLOR = (0, 190, 70)
PRED_COLOR = (220, 40, 40)


class ImageNetTransform:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        array = image.astype(np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--data_name", default="CVOGL_DroneAerial")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--checkpoint", default="saved_models/PCWNet_best.pth")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--save_dir", default="vis")
    parser.add_argument("--topk", type=int, default=7)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> PCWNet:
    path = Path(args.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    saved = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    valid = {field.name for field in fields(PCWNetConfig)}
    values = {key: value for key, value in saved.items() if key in valid}
    values["topk"] = args.topk
    values["query_patch_scales"] = ((0.25, 0.25), (0.50, 0.30), (0.75, 0.40)) if args.data_name == "CVOGL_SVI" else ((0.25, 0.25),)
    model = PCWNet(PCWNetConfig(**values)).to(device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def pixel_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    result = np.asarray(box, dtype=np.float32).copy()
    if np.max(np.abs(result)) <= 1.5:
        result *= np.array([width, height, width, height], dtype=np.float32)
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width - 1)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height - 1)
    return result


def label_box(image: np.ndarray, box: Sequence[float], color: tuple[int, int, int],
              label: str, thickness: int = 3) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = pixel_box(box, width, height).round().astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not label:
        return
    scale = max(min(width, height) / 1000.0, 0.55)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(y1 - th - baseline - 8, 0)
    cv2.rectangle(image, (x1, top), (min(x1 + tw + 10, width - 1),
                  min(top + th + baseline + 8, height - 1)), color, -1)
    cv2.putText(image, label, (x1 + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_candidates(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for index, box in enumerate(boxes):
        color = CANDIDATE_COLORS[index % len(CANDIDATE_COLORS)]
        label_box(canvas, box, color, f"C{index + 1}", 3)
    return canvas


def heatmap_overlay(image: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    heat = probabilities.astype(np.float32)
    heat /= max(float(heat.max()), 1e-12)
    heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_MAGMA)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    alpha = (0.18 + 0.42 * heat)[..., None]
    return np.uint8(np.clip(image * (1.0 - alpha) + colored * alpha, 0, 255))


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    intersection = max(float(right - left), 0.0) * max(float(bottom - top), 0.0)
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def export_sample(model: PCWNet, dataset: RSDataset, index: int,
                  args: argparse.Namespace, device: torch.device) -> Path:
    query, reference, prompt, ground_truth, _, _ = dataset[index]
    with torch.inference_mode():
        outputs = model(query[None].to(device), reference[None].to(device),
                        torch.as_tensor(prompt)[None].to(device))

    query_rgb, reference_rgb = tensor_to_rgb(query), tensor_to_rgb(reference)
    output_dir = Path(args.save_dir)

    prompt_panel = query_rgb.copy()
    py, px = np.unravel_index(np.argmax(prompt), prompt.shape)
    qx = int(round(px * (query_rgb.shape[1] - 1) / max(prompt.shape[1] - 1, 1)))
    qy = int(round(py * (query_rgb.shape[0] - 1) / max(prompt.shape[0] - 1, 1)))
    radius = max(min(query_rgb.shape[:2]) // 60, 5)
    cv2.circle(prompt_panel, (qx, qy), radius + 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(prompt_panel, (qx, qy), radius, GT_COLOR, -1, cv2.LINE_AA)
    save_rgb(output_dir / "01_query_point_prompt.png", prompt_panel)

    ground_truth_panel = reference_rgb.copy()
    label_box(ground_truth_panel, ground_truth, GT_COLOR, "Ground Truth")
    save_rgb(output_dir / "02_satellite_ground_truth.png", ground_truth_panel)

    states = outputs["refinement_states"]
    coarse_boxes = states_to_boxes(states[0])[0].cpu().numpy()
    heat = outputs["anchor_probabilities"][0].cpu().numpy()
    coarse_panel = draw_candidates(heatmap_overlay(reference_rgb, heat), coarse_boxes)
    save_rgb(output_dir / "03_coarse_topk_candidates.png", coarse_panel)

    level_names = list(model.refiners.keys())
    for level, filename in (("layer3", "04_candidates_after_layer3.png"),
                            ("layer1", "05_candidates_after_layer1.png")):
        if level not in level_names:
            raise RuntimeError(f"Checkpoint model has no {level} refinement stage")
        state_index = level_names.index(level) + 1
        boxes = states_to_boxes(states[state_index])[0].cpu().numpy()
        save_rgb(output_dir / filename, draw_candidates(reference_rgb, boxes))

    final_box = outputs["boxes"][0].cpu().numpy()
    selected = int(outputs["selected_indices"][0].item()) + 1
    normalized_gt = np.asarray(ground_truth, dtype=np.float32)
    if np.max(np.abs(normalized_gt)) > 1.5:
        normalized_gt = normalized_gt / np.array(
            [reference_rgb.shape[1], reference_rgb.shape[0],
             reference_rgb.shape[1], reference_rgb.shape[0]], dtype=np.float32
        )
    final_panel = reference_rgb.copy()
    label_box(final_panel, ground_truth, GT_COLOR, "Ground Truth")
    label_box(final_panel, final_box, PRED_COLOR,
              f"Pred C{selected}  IoU={intersection_over_union(final_box, normalized_gt):.3f}")
    save_rgb(output_dir / "06_final_reranked_prediction.png", final_panel)
    return output_dir


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dataset = RSDataset(args.data_root, args.data_name, args.split,
                        transform=ImageNetTransform(), augment=False)
    model = load_model(args, device)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"Sample index {args.index} is outside [0, {len(dataset) - 1}]")
    output_dir = export_sample(model, dataset, args.index, args, device)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
