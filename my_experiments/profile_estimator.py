import os
import sys
import time

import cv2
import numpy as np
import torch

# Add sam-3d-body to python path
current_dir = os.getcwd()
sam3d_root = os.path.join(current_dir, "sam-3d-body")
sys.path.insert(0, sam3d_root)

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body


def benchmark_estimator():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

    print("Loading model...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path, device=device, mhr_path=mhr_path
    )

    # Set smaller input size for speed
    model_cfg.defrost()
    model_cfg.MODEL.IMAGE_SIZE = [384, 384]
    model_cfg.freeze()

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    # Dummy input
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Dummy bbox [x1, y1, x2, y2]
    bboxes = np.array([[500, 200, 1000, 1500]])

    print("Warming up...")
    for _ in range(3):
        estimator.process_one_image(img, bboxes=bboxes, bbox_thr=0.5, use_mask=False)

    print("Benchmarking...")
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        estimator.process_one_image(img, bboxes=bboxes, bbox_thr=0.5, use_mask=False)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    print(f"Average Inference Time: {avg_time:.4f}s")
    print(f"Max Theoretical FPS (Estimator only): {fps:.2f}")


if __name__ == "__main__":
    benchmark_estimator()
