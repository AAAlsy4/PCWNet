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


def parse_args():
    """Parse command-line options for the visualization demo."""
    parser=argparse.ArgumentParser("PCWNet visualization demo")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--data_name", default="CVOGL_DroneAerial")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0, help="image pair index")
    parser.add_argument("--checkpoint", default="saved_models/PCWNet_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="vis")
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


def save_img(path,img):
    """Convert an RGB image to BGR and write it to disk."""
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img)


def draw_box(img, box, color, text=""):
    """Draw a normalized XYXY box and optional label on an image copy."""
    out=img.copy()
    h,w,_=out.shape
    x1,y1,x2,y2=box
    x1=int(x1*w)
    y1=int(y1*h)
    x2=int(x2*w)
    y2=int(y2*h)
    cv2.rectangle(out, (x1,y1), (x2,y2), color, 3)
    cv2.putText(out, text, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return out


def vis_prompt(query_img, prompt_map, save):
    """Visualize the click prompt on the query image."""
    h,w=prompt_map.shape
    cy,cx=np.unravel_index(prompt_map.argmax(), prompt_map.shape)
    # Map back to image coordinates (prompt map is 256x256, image may differ)
    cx_img=int(cx*query_img.shape[1]/w)
    cy_img=int(cy*query_img.shape[0]/h)
    img=query_img.copy()
    cv2.circle(img, (cx_img,cy_img), 12, (0,255,0), 3)
    cv2.circle(img, (cx_img,cy_img), 4, (0,255,0), -1)
    cv2.putText(img, "Prompt", (cx_img+15,cy_img-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    save_img(save, img)


def vis_heatmap(sat, logits, save):
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


def vis_candidates(sat, outputs, save):
    """Draw all candidate boxes with their fused scores."""
    img=sat.copy()
    boxes=outputs["candidate_boxes"][0].cpu().numpy()
    scores=outputs["candidate_scores"][0].cpu().numpy()

    for i,b in enumerate(boxes):
        img=draw_box(img, b, (0,0,255), f"{i}:{scores[i]:.2f}")

    save_img(save, img)


def state_to_box(state):
    """Convert batched center/log-size states to XYXY boxes."""
    center=state[:,:,:2]
    size=torch.exp(state[:,:,2:])

    return torch.cat([center-size/2, center+size/2], dim=-1)


def vis_refinement(sat, outputs, save_dir):
    """Save the candidate box trajectory after each refinement level."""
    colors=[(0,0,255), (0,255,255), (255,165,0), (0,255,0)]
    names=["init","layer3","layer2","layer1"]
    trajectory=sat.copy()

    for i,state in enumerate(outputs["refinement_states"]):
        box=state_to_box(state)[0,0].cpu().numpy()
        trajectory=draw_box(trajectory, box, colors[i], names[i])
        save_img(os.path.join(save_dir, f"0{i+6}_{names[i]}.jpg"), trajectory)

    save_img(os.path.join(save_dir, "05_refinement_trajectory.jpg"), trajectory)


def vis_final(sat, outputs, save):
    """Draw and save the selected final prediction."""
    box=outputs["boxes"][0].cpu().numpy()
    score=float(outputs["scores"][0])
    img=draw_box(sat, box, (0,255,0), f"FINAL:{score:.3f}")
    save_img(save, img)


def main():
    """Load one dataset pair, run PCWNet, and save diagnostic visualizations."""
    args=parse_args()
    reranker = False
    if args.data_name == "CVOGL_SVI":
        reranker = True
    os.makedirs(args.save_dir,exist_ok=True)
    device=torch.device(args.device)

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

    model=PCWNet(PCWNetConfig(
        anchor_score_power=args.anchor_score_power,
        reranker=reranker,
        reranker_weight=args.reranker_weight,
    ))
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs=model(query,reference,prompt)

    # Resize query image to match reference height for side-by-side concatenation
    query_vis_resized = query_vis
    if query_vis.shape[0] != reference_vis.shape[0]:
        new_w = int(query_vis.shape[1] * reference_vis.shape[0] / query_vis.shape[0])
        query_vis_resized = cv2.resize(query_vis, (new_w, reference_vis.shape[0]))
    save_img(os.path.join(args.save_dir, "01_input_pair.jpg"),
        np.concatenate([query_vis_resized, reference_vis], axis=1)
    )

    vis_prompt(query_vis, prompt[0].cpu().numpy(),
        os.path.join(args.save_dir, "02_prompt.jpg")
    )

    vis_heatmap(reference_vis, outputs["anchor_logits"][0],
        os.path.join(args.save_dir, "03_anchor_heatmap.jpg")
    )

    vis_candidates(reference_vis, outputs,
        os.path.join(args.save_dir, "04_topK_candidates_score.jpg")
    )

    vis_refinement(reference_vis, outputs, args.save_dir)

    vis_final(reference_vis, outputs,
        os.path.join(args.save_dir, "10_final_prediction.jpg")
    )


    print("Visualization finished!")


if __name__=="__main__":
    main()
