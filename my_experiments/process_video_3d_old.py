import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add sam-3d-body to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3d_root = os.path.join(current_dir, "../sam-3d-body")
sys.path.insert(0, sam3d_root)
print(f"Added {sam3d_root} to sys.path")

try:
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
    from sam_3d_body.visualization.renderer import Renderer
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from tools.build_detector import HumanDetector
except ImportError as e:
    print(f"Error importing sam-3d-body modules: {e}")
    print(
        f"Please ensure you are running this script from the workspace root or have sam-3d-body correctly installed."
    )
    sys.exit(1)


def main(args):
    # 1. Setup Paths and Output
    video_path = args.video_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 2. Load Models
    print("Loading SAM 3D Body models...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check if checkpoints exist
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found at {args.checkpoint_path}")
        print(
            "Please download it using: hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3"
        )
        return

    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=args.mhr_path
    )

    # Initialize Detector (using ViTDet by default as in demo.py)
    human_detector = None

    det_path = args.detector_path
    if det_path and det_path.lower() == "none":
        print("Skipping detector initialization.")
    else:
        # If path is the broken default and doesn't exist, use empty string to trigger download
        if (
            det_path == "checkpoints/sam-3d-body-dinov3/assets/vitdet_h.py"
            and not os.path.exists(det_path)
        ):
            print(
                "Default detector path not found. Using auto-download for ViTDet weights."
            )
            det_path = ""

        try:
            human_detector = HumanDetector(name="vitdet", device=device, path=det_path)
        except Exception as e:
            print(f"Failed to initialize detector: {e}")
            print(
                "Proceeding without detector (inference will fail if no boxes provided)."
            )

    # Pass human_detector=None to estimator so we can control detection manually
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # 4. Setup Video Writers
    writer_kp = None
    writer_mesh = None
    visualizer = None
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

    if not args.no_vis:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_kp_path = os.path.join(output_dir, f"{video_name}_keypoints.mp4")
        out_mesh_path = os.path.join(output_dir, f"{video_name}_mesh.mp4")

        writer_kp = cv2.VideoWriter(out_kp_path, fourcc, fps, (width, height))
        writer_mesh = cv2.VideoWriter(out_mesh_path, fourcc, fps, (width, height))

        # 5. Setup Visualization Tools
        visualizer = SkeletonVisualizer(line_width=2, radius=5)
        visualizer.set_pose_meta(mhr70_pose_info)

    # 6. Process Loop
    all_keypoints_data = []
    all_mesh_data = []

    frame_idx = 0
    pbar = tqdm(total=total_frames)

    tracked_boxes = None
    skip_interval = args.skip_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detection Logic ---
        # Run detector if:
        # 1. It's the first frame (frame_idx == 0)
        # 2. We are at a refresh interval (frame_idx % (skip + 1) == 0)
        # 3. We don't have tracked boxes yet (e.g. first frame had no detections, try again)

        should_detect = (frame_idx % (skip_interval + 1) == 0) or (
            tracked_boxes is None
        )

        if should_detect and human_detector is not None:
            try:
                # bbox_thr=0.5 is standard
                tracked_boxes = human_detector.run_human_detection(frame, bbox_thr=0.5)
                if len(tracked_boxes) == 0:
                    tracked_boxes = None  # No humans found
            except Exception as e:
                print(f"Detection failed at frame {frame_idx}: {e}")
                tracked_boxes = None

        # --- Inference Logic ---
        # Convert to RGB for SAM 3D Body Estimator
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        outputs = []
        if tracked_boxes is not None:
            outputs = estimator.process_one_image(
                frame_rgb,
                bboxes=tracked_boxes,
                bbox_thr=0.5,
                use_mask=False,
            )

        # Collect Data
        frame_keypoints = []
        frame_vertices = []

        # Visualization buffers
        img_kp_vis = None
        img_mesh_vis = None

        if not args.no_vis:
            img_kp_vis = frame.copy()
            img_mesh_vis = frame.copy()

        # Sort outputs by depth for correct rendering order
        if outputs:
            outputs.sort(key=lambda x: x["pred_cam_t"][2], reverse=True)

        for person_output in outputs:
            # --- Data Collection ---
            # Keypoints 3D
            kp3d = person_output["pred_keypoints_3d"]  # (N_kp, 3)
            frame_keypoints.append(kp3d)

            # Mesh Vertices
            verts = person_output["pred_vertices"]  # (N_verts, 3)
            frame_vertices.append(verts)

            if not args.no_vis:
                # --- Visualization: Keypoints ---
                kp2d = person_output["pred_keypoints_2d"]
                # Add visibility column (ones)
                kp2d_vis = np.concatenate([kp2d, np.ones((kp2d.shape[0], 1))], axis=-1)
                img_kp_vis = visualizer.draw_skeleton(img_kp_vis, kp2d_vis)

                # --- Visualization: Mesh ---
                renderer = Renderer(
                    focal_length=person_output["focal_length"], faces=estimator.faces
                )
                img_mesh_vis = (
                    renderer(
                        person_output["pred_vertices"],
                        person_output["pred_cam_t"],
                        img_mesh_vis,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                    )
                    * 255
                ).astype(np.uint8)

        # Store frame data
        all_keypoints_data.append({"frame_idx": frame_idx, "persons": frame_keypoints})
        all_mesh_data.append(
            {
                "frame_idx": frame_idx,
                "persons": frame_vertices,
                "faces": estimator.faces,  # Store faces once or reference it
            }
        )

        # Write Video Frames
        if not args.no_vis:
            writer_kp.write(img_kp_vis)
            writer_mesh.write(img_mesh_vis)

        pbar.update(1)
        frame_idx += 1

    cap.release()
    if writer_kp:
        writer_kp.release()
    if writer_mesh:
        writer_mesh.release()
    pbar.close()

    # 7. Save Data Files
    kp_data_path = os.path.join(output_dir, f"{video_name}_keypoints_3d.pkl")
    mesh_data_path = os.path.join(output_dir, f"{video_name}_mesh_data.pkl")

    print(f"Saving 3D keypoints data to {kp_data_path}...")
    with open(kp_data_path, "wb") as f:
        pickle.dump(all_keypoints_data, f)

    print(f"Saving Mesh data to {mesh_data_path}...")
    with open(mesh_data_path, "wb") as f:
        # Faces are constant, so we can store them separately or in the first frame
        # But for simplicity, let's just dump the list structure.
        # Note: 'faces' is a numpy array, same for all.
        save_data = {"faces": estimator.faces, "frames": all_mesh_data}
        pickle.dump(save_data, f)

    print("Processing complete!")
    print(f"Outputs saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for SAM 3D Body")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output_3d", help="Directory to save outputs"
    )

    # Default paths assuming standard installation structure or user provided
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/sam-3d-body-dinov3/model.ckpt",
        help="Path to SAM 3D Body checkpoint",
    )
    parser.add_argument(
        "--mhr_path",
        type=str,
        default="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        help="Path to MHR model",
    )
    parser.add_argument(
        "--detector_path",
        type=str,
        default="checkpoints/sam-3d-body-dinov3/assets/vitdet_h.py",
        help="Path to detector config/weights",
    )
    parser.add_argument(
        "--no_vis", action="store_true", help="Disable visualization output (mp4)"
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=0,
        help="Number of frames to skip between detections (uses tracking)",
    )

    args = parser.parse_args()

    # Note on detector path: The demo.py uses 'tools/cascade_mask_rcnn_vitdet_h_75ep.py' usually?
    # Actually demo.py defaults are empty strings and it loads from env or args.
    # The user needs to have the detector weights.
    # Let's try to be helpful with defaults but they might fail if files aren't there.

    main(args)
