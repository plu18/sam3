import argparse
import glob
import os
import pickle
import sys
import time

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

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def process_single_video(video_path, output_dir, estimator, human_detector, args):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {video_name}...")

    # 3. Open Video
    # Explicitly use FFMPEG backend to avoid unnecessary probing of other backends
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    else:
        # Verify backend
        print(f"Successfully opened video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Resize logic
    process_width = width
    process_height = height
    if args.resize_width > 0 and width > args.resize_width:
        scale = args.resize_width / width
        process_width = args.resize_width
        process_height = int(height * scale)
        print(f"Resizing frames to {process_width}x{process_height} for processing.")

    # 4. Setup Video Writers
    writer_kp = None
    writer_mesh = None
    visualizer = None

    if not args.no_vis:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_kp_path = os.path.join(output_dir, f"{video_name}_keypoints.mp4")
        out_mesh_path = os.path.join(output_dir, f"{video_name}_mesh.mp4")

        # Writers use the processed size
        writer_kp = cv2.VideoWriter(
            out_kp_path, fourcc, fps, (process_width, process_height)
        )
        writer_mesh = cv2.VideoWriter(
            out_mesh_path, fourcc, fps, (process_width, process_height)
        )

        # 5. Setup Visualization Tools
        visualizer = SkeletonVisualizer(line_width=2, radius=5)
        visualizer.set_pose_meta(mhr70_pose_info)

    # 6. Process Loop
    all_keypoints_data = []
    all_mesh_data = []

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=video_name)

    tracked_boxes = None
    skip_interval = args.skip_frames

    # Optimization: Use inference mode.
    # Note: Autocast (AMP) is disabled because MHR model uses sparse operations
    # that are not implemented for BFloat16/Float16 on CUDA.
    use_amp = False
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(
        f"Inference optimization: torch.inference_mode() enabled. Autocast disabled (incompatible with MHR sparse ops)."
    )

    start_time = time.time()
    processed_frames_count = 0

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=use_amp
    ):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames_count += 1

            if frame is None:
                print(f"Warning: Frame {frame_idx} is None, skipping.")
                continue

            if frame.size == 0:
                print(f"Warning: Frame {frame_idx} is empty (size 0), skipping.")
                continue

            # Resize frame if needed
            if args.resize_width > 0 and width > args.resize_width:
                frame = cv2.resize(frame, (process_width, process_height))

            # --- Detection Logic ---
            should_detect = (frame_idx % (skip_interval + 1) == 0) or (
                tracked_boxes is None
            )

            if should_detect and human_detector is not None:
                try:
                    # bbox_thr=0.5 is standard
                    # Pass image_size to avoid upscaling in detector
                    det_size = min(process_height, process_width)
                    tracked_boxes = human_detector.run_human_detection(
                        frame, bbox_thr=0.5, image_size=det_size
                    )
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
                    kp2d_vis = np.concatenate(
                        [kp2d, np.ones((kp2d.shape[0], 1))], axis=-1
                    )
                    img_kp_vis = visualizer.draw_skeleton(img_kp_vis, kp2d_vis)

                    # --- Visualization: Mesh ---
                    renderer = Renderer(
                        focal_length=person_output["focal_length"],
                        faces=estimator.faces,
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
            all_keypoints_data.append(
                {"frame_idx": frame_idx, "persons": frame_keypoints}
            )
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

    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = processed_frames_count / total_time if total_time > 0 else 0
    print(f"Processing finished in {total_time:.2f}s. Average FPS: {avg_fps:.2f}")

    # 7. Save Data Files
    kp_data_path = os.path.join(output_dir, f"{video_name}_keypoints_3d.pkl")
    mesh_data_path = os.path.join(output_dir, f"{video_name}_mesh_data.pkl")

    print(f"Saving 3D keypoints data to {kp_data_path}...")
    with open(kp_data_path, "wb") as f:
        pickle.dump(all_keypoints_data, f)

    print(f"Saving Mesh data to {mesh_data_path}...")
    with open(mesh_data_path, "wb") as f:
        save_data = {"faces": estimator.faces, "frames": all_mesh_data}
        pickle.dump(save_data, f)

    print(f"Finished processing {video_name}")


def main(args):
    # 1. Setup Paths and Output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Models
    print("Loading SAM 3D Body models...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Optimization: Enable CuDNN benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Inference optimization: torch.backends.cudnn.benchmark = True")

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

    # Override model input size if requested
    if args.model_input_size != 512:
        print(f"Overriding model input size to {args.model_input_size}")
        model_cfg.defrost()
        model_cfg.MODEL.IMAGE_SIZE = [args.model_input_size, args.model_input_size]
        model_cfg.freeze()

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

    # 3. Determine input videos
    video_paths = []
    if args.input_dir:
        if os.path.isdir(args.input_dir):
            # Find all video files
            extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
            video_paths = []
            for ext in extensions:
                video_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            video_paths.sort()
            print(f"Found {len(video_paths)} videos in {args.input_dir}")
        else:
            print(f"Input directory not found: {args.input_dir}")
            return
    elif args.video_path:
        video_paths = [args.video_path]
    else:
        print("Please provide either --video_path or --input_dir")
        return

    # 4. Process Loop
    for video_path in video_paths:
        try:
            process_single_video(
                video_path, output_dir, estimator, human_detector, args
            )
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback

            traceback.print_exc()

    print("All videos processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for SAM 3D Body")
    parser.add_argument(
        "--video_path", type=str, help="Path to input video (single file)"
    )
    parser.add_argument(
        "--input_dir", type=str, help="Path to input directory (batch process)"
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
    parser.add_argument(
        "--resize_width",
        type=int,
        default=640,
        help="Resize video width for processing (0 to disable)",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=512,
        help="Input size for the SAM 3D Body model (default: 512). Reduce to 384 or 256 for speed.",
    )

    args = parser.parse_args()

    main(args)
