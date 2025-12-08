import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def find_cached_checkpoint():
    # Try to find sam3.pt in the default HF cache directory
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--sam3/snapshots"
    )
    if os.path.exists(cache_dir):
        snapshots = glob.glob(os.path.join(cache_dir, "*"))
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            ckpt_path = os.path.join(latest_snapshot, "sam3.pt")
            if os.path.exists(ckpt_path):
                print(f"Found cached checkpoint at: {ckpt_path}")
                return ckpt_path
    return None


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -4)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def save_visualization(image, masks, boxes, scores, prompt, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Filter by score if needed, but let's show all returned
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if score > 0.5:  # Only show confident predictions
            mask_np = mask.cpu().numpy()
            box_np = box.cpu().numpy()
            score_val = score.item()

            show_mask(mask_np, plt.gca(), random_color=True)
            show_box(box_np, plt.gca())
            # Add score text
            plt.text(
                box_np[0],
                box_np[1],
                f"{score_val:.2f}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="green", alpha=0.5),
            )

    plt.title(f"Prompt: {prompt}")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    # 1. Load Model
    print("Building SAM3 image model...")
    cached_ckpt = find_cached_checkpoint()
    load_from_hf = True
    ckpt_path = None

    if cached_ckpt:
        print("Using local cached checkpoint.")
        load_from_hf = False
        ckpt_path = cached_ckpt

    try:
        model = build_sam3_image_model(
            load_from_HF=load_from_hf, checkpoint_path=ckpt_path
        )
    except Exception as e:
        print(f"Error building model: {e}")
        return

    processor = Sam3Processor(model)

    # 2. Load Image
    image_path = "/workspaces/sctsam3/assets/images/truck.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")

    # 3. Set Image
    inference_state = processor.set_image(image)

    # 4. Run Inference with Text Prompts
    prompts = ["truck", "wheel", "window", "door"]

    for prompt in prompts:
        print(f"Running inference for prompt: '{prompt}'")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        output_file = f"/workspaces/sctsam3/my_experiments/output/result_{prompt}.png"
        save_visualization(image, masks, boxes, scores, prompt, output_file)

    # 5. Run Inference with Box Prompt (Simulated)
    # Let's use the box from the "truck" prompt as a new prompt
    print("Running inference with Box Prompt (using 'truck' box)...")
    # Re-run truck to get the box
    output = processor.set_text_prompt(state=inference_state, prompt="truck")
    truck_boxes = output["boxes"]
    truck_scores = output["scores"]

    if len(truck_boxes) > 0:
        # Pick the highest scoring box
        best_idx = torch.argmax(truck_scores)
        best_box = truck_boxes[best_idx]  # [x0, y0, x1, y1]

        # Convert to [cx, cy, w, h] normalized
        w_img, h_img = image.size
        x0, y0, x1, y1 = best_box
        w_box = x1 - x0
        h_box = y1 - y0
        cx = x0 + w_box / 2
        cy = y0 + h_box / 2

        norm_box = [cx / w_img, cy / h_img, w_box / w_img, h_box / h_img]

        # Reset prompts before adding geometric prompt?
        # The API says "The image needs to be set, but not necessarily the text prompt."
        # But `add_geometric_prompt` appends. Let's reset first.
        processor.reset_all_prompts(inference_state)

        # Note: add_geometric_prompt expects a list/tensor for box?
        # Docstring: "box: List ... [center_x, center_y, width, height]"
        print(f"Using box prompt: {norm_box}")
        output_box = processor.add_geometric_prompt(
            box=norm_box, label=True, state=inference_state
        )

        masks_box = output_box["masks"]
        boxes_box = output_box["boxes"]
        scores_box = output_box["scores"]

        output_file_box = (
            "/workspaces/sctsam3/my_experiments/output/result_box_prompt.png"
        )
        save_visualization(
            image,
            masks_box,
            boxes_box,
            scores_box,
            "Box Prompt (from truck)",
            output_file_box,
        )
    else:
        print("No truck box found to use for box prompt test.")


if __name__ == "__main__":
    main()
