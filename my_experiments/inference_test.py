import glob
import os

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
        # Find the most recent snapshot
        snapshots = glob.glob(os.path.join(cache_dir, "*"))
        if snapshots:
            # Sort by modification time just in case, though usually there's one active
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            ckpt_path = os.path.join(latest_snapshot, "sam3.pt")
            if os.path.exists(ckpt_path):
                print(f"Found cached checkpoint at: {ckpt_path}")
                return ckpt_path
    return None


# Create a dummy image for testing since we don't have one
def create_dummy_image(path):
    img = Image.new("RGB", (512, 512), color="red")
    img.save(path)
    return path


def main():
    print("Building SAM3 image model...")

    cached_ckpt = find_cached_checkpoint()
    load_from_hf = True
    ckpt_path = None

    if cached_ckpt:
        print("Using local cached checkpoint to avoid re-download/auth issues.")
        load_from_hf = False
        ckpt_path = cached_ckpt
    else:
        print(
            "No cached checkpoint found. Will attempt to download from HF (requires auth)."
        )

    try:
        model = build_sam3_image_model(
            load_from_HF=load_from_hf, checkpoint_path=ckpt_path
        )
    except Exception as e:
        print(f"Error building model: {e}")
        return

    processor = Sam3Processor(model)

    # Determine output directory (same as script location)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(output_dir, "test_image.jpg")
    result_path = os.path.join(output_dir, "inference_results.txt")

    create_dummy_image(image_path)
    print(f"Created dummy image at {image_path}")

    try:
        image = Image.open(image_path)
        print("Processing image...")
        inference_state = processor.set_image(image)

        prompt = "a red square"
        print(f"Prompting with: '{prompt}'")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        print("Inference successful!")
        print(f"Masks shape: {masks.shape}")
        print(f"Boxes shape: {boxes.shape}")
        print(f"Scores shape: {scores.shape}")

        # Save results to file
        with open(result_path, "w") as f:
            f.write("Inference successful!\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Masks shape: {masks.shape}\n")
            f.write(f"Boxes shape: {boxes.shape}\n")
            f.write(f"Scores shape: {scores.shape}\n")
        print(f"Results saved to {result_path}")

    except Exception as e:
        print(f"Inference failed: {e}")
        # Write failure to result file as well
        with open(result_path, "w") as f:
            f.write(f"Inference failed: {e}\n")

    # Removed cleanup to persist the test image


if __name__ == "__main__":
    main()
