import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.PCWNet import PCWNet, PCWNetConfig
from utils.data_loader import RSDataset


GROUND_TRUTH_COLOR=(0,255,0)
CANDIDATE_COLOR=(0,0,255)
PREDICTION_COLOR=(255,0,0)
PROMPT_COLOR=(0,255,0)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed=int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the visualization demo."""
    parser=argparse.ArgumentParser("PCWNet visualization demo")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--data_name", default="CVOGL_DroneAerial")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0, help="image pair index")
    parser.add_argument("--checkpoint", default="saved_models/PCWNet_best.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_dir", default="vis")
    parser.add_argument("--topk", type=positive_int, default=5,
        help="number of candidate boxes to refine")
    parser.add_argument("--anchor_score_power", type=float, default=1.0)
    parser.add_argument("--reranker_weight", type=float, default=1.0)

    return parser.parse_args()


class ImageNetTransform:
    """Convert images to ImageNet-normalized tensors and restore them for display."""

    def __init__(self):
        """Store ImageNet normalization statistics."""

        self.mean=torch.tensor([0.485,0.456,0.406])[:,None,None]
        self.std=torch.tensor([0.229,0.224,0.225])[:,None,None]

    def __call__(self, image):
        """Normalize an RGB image array and return a channel-first tensor."""

        tensor=torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1)
        tensor=tensor.float().div(255.0)
        return (tensor-self.mean)/self.std

    def denormalize(self, img_tensor):

        """Reverse normalization: img * std + mean recovers [0,1], then scale to uint8 [0,255]."""
        img=img_tensor*self.std+self.mean
        img=img.clamp(0,1)
        img=(img*255).byte()
        return img


def save_img(path,img) -> None:
    """Convert an RGB image to BGR and write it to disk."""
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img)


def draw_box(img, box, color) -> np.ndarray:
    """Draw a normalized XYXY box on an image copy."""
    out=img.copy()
    h,w,_=out.shape
    x1,y1,x2,y2=np.clip(np.asarray(box), 0.0, 1.0)
    x1=int(x1*w)
    y1=int(y1*h)
    x2=int(x2*w)
    y2=int(y2*h)
    cv2.rectangle(out, (x1,y1), (x2,y2), color, 3)

    return out


def draw_boxes(img, boxes, color) -> np.ndarray:
    """Draw a collection of normalized XYXY boxes in one color."""
    out=img.copy()
    for box in boxes:
        out=draw_box(out, box, color)
    return out


def add_legend(img, entries) -> np.ndarray:
    """Add a compact box-color legend to the top-right image corner."""
    out=img.copy()
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=0.55
    thickness=1
    padding=12
    swatch_size=20
    row_height=30
    text_width=max(
        cv2.getTextSize(label, font, font_scale, thickness)[0][0]
        for label,_ in entries
    )
    legend_width=padding*3+swatch_size+text_width
    legend_height=padding*2+row_height*len(entries)
    x1=max(out.shape[1]-legend_width-padding, 0)
    y1=padding
    x2=min(x1+legend_width, out.shape[1]-1)
    y2=min(y1+legend_height, out.shape[0]-1)

    overlay=out.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (255,255,255), -1)
    out=cv2.addWeighted(overlay, 0.85, out, 0.15, 0)
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,0), 1)

    for index,(label,color) in enumerate(entries):
        row_y=y1+padding+index*row_height
        swatch_y=row_y+(row_height-swatch_size)//2
        swatch_x=x1+padding
        cv2.rectangle(
            out,
            (swatch_x,swatch_y),
            (swatch_x+swatch_size,swatch_y+swatch_size),
            color,
            3,
        )
        text_y=row_y+(row_height+10)//2
        cv2.putText(
            out,
            label,
            (swatch_x+swatch_size+padding,text_y),
            font,
            font_scale,
            (0,0,0),
            thickness,
            cv2.LINE_AA,
        )

    return out


def normalize_box(box, image) -> np.ndarray:
    """Normalize a pixel-space XYXY box against an image array."""
    height,width=image.shape[:2]
    scale=np.array([width,height,width,height], dtype=np.float32)
    return np.clip(np.asarray(box, dtype=np.float32)/scale, 0.0, 1.0)


def vis_prompt(query_img, prompt_map, save) -> None:
    """Visualize the click prompt on the query image."""
    h,w=prompt_map.shape
    cy,cx=np.unravel_index(prompt_map.argmax(), prompt_map.shape)
    # Map back to image coordinates (prompt map is 256x256, image may differ)
    cx_img=int(cx*query_img.shape[1]/w)
    cy_img=int(cy*query_img.shape[0]/h)
    img=query_img.copy()
    cv2.circle(img, (cx_img,cy_img), 12, PROMPT_COLOR, 3)
    cv2.circle(img, (cx_img,cy_img), 4, PROMPT_COLOR, -1)
    save_img(save, img)


def vis_heatmap(sat, logits, save) -> None:
    """Overlay normalized anchor logits on the reference image."""
    heatmap=logits.cpu().numpy()
    heatmap=np.exp(heatmap-heatmap.max())
    heatmap/=heatmap.max()
    heatmap=cv2.resize(heatmap, (sat.shape[1], sat.shape[0]))
    plt.figure(figsize=(8,8))
    plt.imshow(sat)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


def vis_candidates(sat, outputs, target_box, save) -> None:
    """Draw all final candidate boxes and the dataset ground truth."""
    boxes=outputs["candidate_boxes"][0].cpu().numpy()
    img=draw_boxes(sat, boxes, CANDIDATE_COLOR)
    img=draw_box(img, target_box, GROUND_TRUTH_COLOR)
    img=add_legend(img, [
        ("Candidates", CANDIDATE_COLOR),
        ("Ground truth", GROUND_TRUTH_COLOR),
    ])

    save_img(save, img)


def state_to_box(state) -> torch.Tensor:
    """Convert batched center/log-size states to XYXY boxes."""
    center=state[:,:,:2]
    size=torch.exp(state[:,:,2:])

    return torch.cat([center-size/2, center+size/2], dim=-1)


def vis_refinement(sat, outputs, target_box, save_dir) -> None:
    """Save cumulative trajectories for every candidate refinement state."""
    colors=[(0,0,255), (0,255,255), (255,165,0), (255,0,255)]
    names=["init","layer3","layer2","layer1"]
    labels=["Initial","After layer3","After layer2","After layer1"]
    trajectory=sat.copy()
    legend_entries=[]

    for i,state in enumerate(outputs["refinement_states"]):
        boxes=state_to_box(state)[0].cpu().numpy()
        trajectory=draw_boxes(trajectory, boxes, colors[i])
        legend_entries.append((labels[i], colors[i]))
        stage_img=draw_box(trajectory, target_box, GROUND_TRUTH_COLOR)
        stage_img=add_legend(stage_img, legend_entries+[
            ("Ground truth", GROUND_TRUTH_COLOR),
        ])
        save_img(os.path.join(save_dir, f"{i+7:02d}_{names[i]}.jpg"), stage_img)

    trajectory=draw_box(trajectory, target_box, GROUND_TRUTH_COLOR)
    trajectory=add_legend(trajectory, legend_entries+[
        ("Ground truth", GROUND_TRUTH_COLOR),
    ])
    save_img(os.path.join(save_dir, "06_refinement_trajectory.jpg"), trajectory)


def vis_final(sat, outputs, target_box, save) -> None:
    """Draw the selected final prediction against the dataset ground truth."""
    box=outputs["boxes"][0].cpu().numpy()
    img=draw_box(sat, box, PREDICTION_COLOR)
    img=draw_box(img, target_box, GROUND_TRUTH_COLOR)
    img=add_legend(img, [
        ("Prediction", PREDICTION_COLOR),
        ("Ground truth", GROUND_TRUTH_COLOR),
    ])
    save_img(save, img)


def resolve_device(name: str) -> torch.device:
    """Resolve an explicit device name or choose CUDA when available."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    """Load one dataset pair, run PCWNet, and save diagnostic visualizations."""
    args=parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    device = resolve_device(args.device)

    dataset=RSDataset(
        data_root=args.data_root,
        data_name=args.data_name,
        split_name=args.split,
        img_size=1024,
        transform=ImageNetTransform(),
        augment=False
    )

    query,reference,prompt,boxes,_,_=dataset[args.index]
    query=query.unsqueeze(0).to(device)
    reference=reference.unsqueeze(0).to(device)
    prompt=torch.from_numpy(prompt).unsqueeze(0).to(device)

    transform=ImageNetTransform()
    query_vis=transform.denormalize(query[0].cpu()).permute(1,2,0).numpy()
    reference_vis=transform.denormalize(reference[0].cpu()).permute(1,2,0).numpy()
    target_box=normalize_box(boxes, reference_vis)

    model=PCWNet(PCWNetConfig(
        topk=args.topk,
        anchor_score_power=args.anchor_score_power,
        reranker_weight=args.reranker_weight,
    ))
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs=model(query,reference,prompt)

    save_img(os.path.join(args.save_dir, "01_query_image.jpg"), query_vis)
    save_img(os.path.join(args.save_dir, "02_reference_image.jpg"), reference_vis)

    vis_prompt(query_vis, prompt[0].cpu().numpy(),
        os.path.join(args.save_dir, "03_prompt.jpg")
    )

    vis_heatmap(reference_vis, outputs["anchor_logits"][0],
        os.path.join(args.save_dir, "04_anchor_heatmap.jpg")
    )

    vis_candidates(reference_vis, outputs, target_box,
        os.path.join(args.save_dir, "05_topK_candidates.jpg")
    )

    vis_refinement(reference_vis, outputs, target_box, args.save_dir)

    vis_final(reference_vis, outputs, target_box,
        os.path.join(args.save_dir, "11_final_prediction.jpg")
    )


    print("Visualization finished!")


if __name__=="__main__":
    main()
