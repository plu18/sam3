import argparse
import os
import sys
import time
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from unittest.mock import MagicMock

# --- MOCK DETECTRON2 ---
# SAM 3D Body imports detectron2 for ViTDet, but we are using YOLO/FasterRCNN.
# We mock it to bypass the import error since we don't use the internal detector.
try:
    import detectron2
except ImportError:
    print("Detectron2 not found. Mocking it to bypass dependency...")
    sys.modules["detectron2"] = MagicMock()
    sys.modules["detectron2.config"] = MagicMock()
    sys.modules["detectron2.modeling"] = MagicMock()
    sys.modules["detectron2.checkpoint"] = MagicMock()
    sys.modules["detectron2.data"] = MagicMock()
    sys.modules["detectron2.engine"] = MagicMock()
    sys.modules["detectron2.structures"] = MagicMock()
    sys.modules["detectron2.utils"] = MagicMock()
    sys.modules["detectron2.utils.logger"] = MagicMock()
    sys.modules["detectron2.layers"] = MagicMock()

# Add workspace root to path to find sam-3d-body
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
SAM3D_ROOT = os.path.join(WORKSPACE_ROOT, "sam-3d-body")
sys.path.insert(0, SAM3D_ROOT)

# Import Renderer
try:
    from sam_3d_body.visualization.renderer import Renderer
except ImportError:
    print("Could not import Renderer. Visualization will be skipped.")
    Renderer = None

try:
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
except ImportError as e:
    print(f"Error importing SAM 3D Body: {e}")
    sys.exit(1)

# --- Helper: Project 3D to 2D ---
def project_points(points_3d, cam_t, focal_length, img_h, img_w):
    """
    Project 3D points to 2D image plane.
    points_3d: (N, 3)
    cam_t: (3,)
    focal_length: float
    """
    # Camera coordinate system in SAM3D:
    # X-right, Y-down, Z-forward (usually)
    # But Renderer flips X for pyrender.
    # Let's assume standard pinhole projection.
    
    # Translate
    points_cam = points_3d + cam_t
    
    # Project
    # u = f * x / z + cx
    # v = f * y / z + cy
    
    cx = img_w / 2.0
    cy = img_h / 2.0
    
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    
    u = (focal_length * x / z) + cx
    v = (focal_length * y / z) + cy
    
    return np.stack([u, v], axis=1)

# --- 1. Modular Detector ---
class BaseDetector:
    def detect(self, image):
        raise NotImplementedError

class YOLODetector(BaseDetector):
    def __init__(self, model_name="yolov8l.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print(f"Loaded YOLOv8 model: {model_name}")
        except ImportError:
            raise ImportError("Ultralytics not installed")

    def detect(self, image):
        # YOLO expects BGR or RGB. It handles it.
        results = self.model(image, verbose=False, classes=[0]) # 0 is person
        boxes = []
        for r in results:
            boxes.extend(r.boxes.xyxy.cpu().numpy())
        return np.array(boxes)

class FasterRCNNDetector(BaseDetector):
    def __init__(self, device="cuda"):
        import torchvision
        print("Loading FasterRCNN (Fallback)...")
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()

    def detect(self, image):
        # Image is BGR (OpenCV), convert to RGB tensor 0-1
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        # Filter for person (label 1) and score > 0.5
        boxes = []
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if label == 1 and score > 0.7:
                boxes.append(box.cpu().numpy())
        
        return np.array(boxes)

# --- 2. Simple Tracker ---
class IoUTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = {} # id -> box
        self.next_id = 1
        self.iou_threshold = iou_threshold

    def update(self, boxes):
        # Simple greedy assignment
        if len(boxes) == 0:
            return {}
        
        new_tracks = {}
        used_boxes = set()
        
        # 1. Match existing tracks
        for tid, old_box in self.tracks.items():
            best_iou = 0
            best_idx = -1
            
            for i, new_box in enumerate(boxes):
                if i in used_boxes: continue
                iou = self.compute_iou(old_box, new_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou > self.iou_threshold:
                new_tracks[tid] = boxes[best_idx]
                used_boxes.add(best_idx)
        
        # 2. Create new tracks
        for i, new_box in enumerate(boxes):
            if i not in used_boxes:
                new_tracks[self.next_id] = new_box
                self.next_id += 1
                
        self.tracks = new_tracks
        return new_tracks

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

# --- 3. Interpolation Utils ---
def interpolate_pose(pose1, pose2, alpha):
    # pose: (N, 3, 3) rotation matrices
    # Flatten to (N, 9) or keep as is? Scipy needs (N, 4) quats or (N, 3, 3)
    # We assume pose is (N_joints, 3, 3)
    
    n_joints = pose1.shape[0]
    r1 = R.from_matrix(pose1)
    r2 = R.from_matrix(pose2)
    
    # Slerp
    key_times = [0, 1]
    slerp = Slerp(key_times, R.from_matrix(np.stack([pose1, pose2], axis=0).reshape(-1, 3, 3)))
    # This is tricky with multiple joints. 
    # Easier: Loop over joints (slow) or reshape
    
    # Vectorized Slerp
    quats1 = r1.as_quat()
    quats2 = r2.as_quat()
    
    # Simple linear interp of quaternions + normalize (NLERP) is often enough for small steps
    # But let's try to use scipy slerp properly if possible, or just NLERP
    
    # NLERP implementation
    dot = np.sum(quats1 * quats2, axis=1, keepdims=True)
    # Flip sign if dot < 0 for shortest path
    q2_fixed = np.where(dot < 0, -quats2, quats2)
    
    q_interp = (1 - alpha) * quats1 + alpha * q2_fixed
    q_interp = q_interp / np.linalg.norm(q_interp, axis=1, keepdims=True)
    
    return R.from_quat(q_interp).as_matrix()

def interpolate_shape(shape1, shape2, alpha):
    return (1 - alpha) * shape1 + alpha * shape2

# --- 4. Main Pipeline ---
def main(args):
    # A. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # B. Load Detector
    try:
        detector = YOLODetector()
    except ImportError:
        print("YOLO not found, falling back to FasterRCNN (slower but works)")
        detector = FasterRCNNDetector(device=device)
        
    tracker = IoUTracker()
    
    # C. Load SAM 3D
    print("Loading SAM 3D Body...")
    model, model_cfg = load_sam_3d_body(args.checkpoint, device=device, mhr_path=args.mhr)
    estimator = SAM3DBodyEstimator(model, model_cfg, human_detector=None)
    
    # Initialize Skeleton Visualizer
    skel_vis = SkeletonVisualizer(
        kpt_color=(0, 0, 255), # Red in BGR
        link_color=(255, 255, 255), # White
        radius=2,
        line_width=2
    )
    skel_vis.set_pose_meta(mhr70_pose_info)
    
    # D. Video Loop
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # Data Storage
    # track_id -> { frame_idx: { 'pose': ..., 'shape': ..., 'trans': ... } }
    track_data = {}
    
    # Load Segmentation
    seg_path = os.path.join(CURRENT_DIR, "mhr_segmentation.npy")
    vertex_groups = None
    if os.path.exists(seg_path):
        print(f"Loading segmentation from {seg_path}")
        vertex_groups = np.load(seg_path)
        
        # Define Colors (RGB)
        # 0: Torso (Purple)
        # 1: Upper Arm (Green)
        # 2: Forearm (Cyan)
        # 3: Hand (Orange)
        # 4: Thigh (Blue)
        # 5: Calf (Yellow)
        # 6: Foot (Pink)
        # 7: Head (Grey)
        
        GROUP_COLORS = np.array([
            [0.5, 0.0, 0.5], # 0
            [0.0, 0.8, 0.0], # 1
            [0.0, 0.8, 0.8], # 2
            [1.0, 0.5, 0.0], # 3
            [0.0, 0.0, 1.0], # 4
            [1.0, 1.0, 0.0], # 5
            [1.0, 0.0, 1.0], # 6
            [0.7, 0.7, 0.7], # 7
        ])
    else:
        print("Segmentation file not found. Using uniform color.")

    # Buffers for interpolation
    # track_id -> { 'last_keyframe_idx': int, 'last_data': dict }
    track_state = {}
    
    pbar = tqdm(total=total_frames)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect & Track
        boxes = detector.detect(frame)
        tracks = tracker.update(boxes) # {tid: [x1,y1,x2,y2]}
        
        # 2. Process Tracks
        frame_vis = frame.copy()
        
        for tid, box in tracks.items():
            # Draw Box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_vis, f"ID {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 3. Keyframe Logic
            is_keyframe = (frame_idx % args.skip == 0)
            
            # If new track, force keyframe
            if tid not in track_state:
                is_keyframe = True
                
            current_data = None
            
            if is_keyframe:
                # Run SAM 3D
                # Prepare input: RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Run on single box
                outputs = estimator.process_one_image(frame_rgb, bboxes=np.array([box]), bbox_thr=0.0)
                
                if outputs:
                    out = outputs[0] # Assume 1 person per box
                    current_data = {
                        'pose': out.get('pred_rotmat', np.eye(3)), # Need to check key name
                        'shape': out.get('pred_betas', np.zeros(10)),
                        'trans': out.get('pred_cam_t', np.zeros(3)),
                        'verts': out['pred_vertices'],
                        'joints': out['pred_keypoints_3d'],
                        'focal_length': out.get('focal_length', 5000)
                    }
                    # Update State
                    track_state[tid] = {
                        'last_keyframe_idx': frame_idx,
                        'last_data': current_data
                    }
            else:
                # Interpolate
                if tid in track_state:
                    last_idx = track_state[tid]['last_keyframe_idx']
                    last_data = track_state[tid]['last_data']
                    
                    # For real-time, we can only hold the last frame (Zero-Order Hold)
                    # For offline, we would need to look ahead. 
                    # HERE: We implement Zero-Order Hold for simplicity in this pass
                    # To do Linear Interp, we need a 2-pass approach or a buffer delay.
                    # Let's stick to Zero-Order Hold + Box Adjustment for now to prove speed.
                    
                    current_data = last_data.copy()
                    # Adjust translation based on box center difference? 
                    # Complex without depth. Keep as is.
            
            # Store Data
            if current_data:
                if tid not in track_data: track_data[tid] = []
                track_data[tid].append({
                    'frame_idx': frame_idx,
                    'data': current_data
                })
                
                # Visualize Mesh (Muscle Style)
                if Renderer is not None:
                    try:
                        focal_length = current_data.get('focal_length', 5000)
                        renderer = Renderer(focal_length=focal_length, faces=estimator.faces)
                        
                        # Prepare Colors
                        if vertex_groups is not None:
                            mesh_colors = GROUP_COLORS[vertex_groups]
                        else:
                            mesh_colors = (0.8, 0.2, 0.2)

                        # 1. Render Mesh (Muscle Colors, Semi-Transparent)
                        rendered = renderer(
                            current_data['verts'],
                            current_data['trans'],
                            frame_vis,
                            mesh_base_color=mesh_colors, 
                            scene_bg_color=(1, 1, 1),
                            alpha=0.8
                        )
                        frame_vis = (rendered * 255).astype(np.uint8)
                        
                        # 2. Render Skeleton (Colored by Part)
                        # Project 3D joints to 2D
                        joints_3d = current_data['joints'] # (70, 3)
                        cam_t = current_data['trans']
                        h, w = frame_vis.shape[:2]
                        
                        kpts_2d = project_points(joints_3d, cam_t, focal_length, h, w)
                        
                        # Define Bones and Colors
                        # Colors in BGR (OpenCV)
                        C_TORSO = (128, 0, 128) # Purple
                        C_ARM_U = (0, 200, 0)   # Green
                        C_ARM_L = (200, 200, 0) # Cyan-ish
                        C_LEG_U = (200, 0, 0)   # Blue
                        C_LEG_L = (0, 200, 200) # Yellow
                        C_HEAD = (128, 128, 128)# Grey
                        
                        # MHR70 indices
                        # Head: 0-4, 69(neck)
                        # Shoulders: 5, 6
                        # Elbows: 7, 8
                        # Wrists: 63, 41 (from previous code, need to verify)
                        # Hips: 9, 10
                        # Knees: 11, 12
                        # Ankles: 13, 14
                        
                        # Let's use a simplified bone list with colors
                        bones_colored = [
                            # Head
                            (0, 1, C_HEAD), (0, 2, C_HEAD), (1, 2, C_HEAD),
                            (0, 69, C_HEAD),
                            # Torso
                            (69, 5, C_TORSO), (69, 6, C_TORSO), # Neck to Shoulders
                            (69, 9, C_TORSO), (69, 10, C_TORSO), # Neck to Hips (Spine approx)
                            (5, 9, C_TORSO), (6, 10, C_TORSO), # Side body
                            (9, 10, C_TORSO), # Hips
                            # Arms (Left)
                            (5, 7, C_ARM_U), # Shoulder -> Elbow
                            (7, 63, C_ARM_L), # Elbow -> Wrist (63 is L Wrist in MHR70?)
                            # Arms (Right)
                            (6, 8, C_ARM_U),
                            (8, 41, C_ARM_L), # 41 is R Wrist?
                            # Legs (Left)
                            (9, 11, C_LEG_U), # Hip -> Knee
                            (11, 13, C_LEG_L), # Knee -> Ankle
                            # Legs (Right)
                            (10, 12, C_LEG_U),
                            (12, 14, C_LEG_L),
                        ]
                        
                        # Draw Joints
                        for j in kpts_2d:
                            cv2.circle(frame_vis, (int(j[0]), int(j[1])), 3, (255, 255, 255), -1)
                            
                        # Draw Bones
                        for i1, i2, color in bones_colored:
                            if i1 < len(kpts_2d) and i2 < len(kpts_2d):
                                pt1 = (int(kpts_2d[i1][0]), int(kpts_2d[i1][1]))
                                pt2 = (int(kpts_2d[i2][0]), int(kpts_2d[i2][1]))
                                cv2.line(frame_vis, pt1, pt2, color, 2)
                        
                    except Exception as e:
                        # print(f"Render error: {e}")
                        pass

        out_vid.write(frame_vis)
        pbar.update(1)
        frame_idx += 1
        
    cap.release()
    out_vid.release()
    
    # Save PKL
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(track_data, f)
    print(f"Saved data to {args.output_pkl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="output_fast.mp4")
    parser.add_argument("--output_pkl", type=str, default="output_fast.pkl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr", type=str, default="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame with SAM3")
    args = parser.parse_args()
    
    main(args)
