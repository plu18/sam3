
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock

# Mock detectron2
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

# Add workspace root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
SAM3D_ROOT = os.path.join(WORKSPACE_ROOT, "sam-3d-body")
sys.path.insert(0, SAM3D_ROOT)

from sam_3d_body import load_sam_3d_body

def extract_segmentation():
    checkpoint_path = "checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    device = "cpu" # CPU is enough for this

    print(f"Loading model from {checkpoint_path}...")
    model, _ = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    
    mhr = model.head_pose.mhr
    char = mhr.character_torch
    lbs = char.linear_blend_skinning
    skel = char.skeleton
    
    # Get skin weights
    # Sparse representation: weights, indices, vert_indices
    weights = lbs.skin_weights_flattened.detach().numpy()
    joint_indices = lbs.skin_indices_flattened.detach().numpy()
    vert_indices = lbs.vert_indices_flattened.detach().numpy()
    
    num_verts = char.mesh.rest_vertices.shape[0]
    print(f"Num vertices: {num_verts}")
    
    # Reconstruct dense weights (or just find max)
    # We want dominant joint for each vertex
    # Initialize with -1
    vertex_dominant_joint = np.full(num_verts, -1, dtype=int)
    vertex_max_weight = np.zeros(num_verts, dtype=float)
    
    print("Computing dominant joints...")
    for i in range(len(weights)):
        v_idx = vert_indices[i]
        w = weights[i]
        j_idx = joint_indices[i]
        
        if w > vertex_max_weight[v_idx]:
            vertex_max_weight[v_idx] = w
            vertex_dominant_joint[v_idx] = j_idx
            
    joint_names = skel.joint_names
    print(f"Num joints: {len(joint_names)}")
    
    # Define groups
    # 0: Torso/Core (Abs, Back, Pecs)
    # 1: Upper Arm (Delts, Biceps, Triceps)
    # 2: Forearm
    # 3: Hand
    # 4: Thigh (Glutes, Quads, Hams)
    # 5: Calf
    # 6: Foot
    # 7: Head/Neck
    
    vertex_groups = np.zeros(num_verts, dtype=int)
    
    for v_idx in range(num_verts):
        j_idx = vertex_dominant_joint[v_idx]
        if j_idx == -1:
            continue
            
        j_name = joint_names[j_idx]
        
        if 'head' in j_name or 'neck' in j_name or 'jaw' in j_name or 'eye' in j_name or 'teeth' in j_name or 'tongue' in j_name:
            group = 7
        elif 'hand' in j_name or 'wrist' in j_name or 'pinky' in j_name or 'ring' in j_name or 'middle' in j_name or 'index' in j_name or 'thumb' in j_name:
            # Exclude wrist_twist if it belongs to forearm? Usually wrist is hand.
            if 'twist' in j_name and 'wrist' in j_name:
                group = 2 # Forearm
            else:
                group = 3
        elif 'lowarm' in j_name: # Forearm
            group = 2
        elif 'uparm' in j_name: # Upper Arm
            group = 1
        elif 'clavicle' in j_name: # Shoulder/Torso
            group = 0 # Treat clavicle as Torso/Shoulder base
        elif 'spine' in j_name or 'root' in j_name or 'body' in j_name: # Torso
            group = 0
        elif 'foot' in j_name or 'talocrural' in j_name or 'subtalar' in j_name or 'transversetarsal' in j_name or 'ball' in j_name: # Foot
            group = 6
        elif 'lowleg' in j_name: # Calf
            group = 5
        elif 'upleg' in j_name: # Thigh
            group = 4
        else:
            print(f"Unclassified joint: {j_name}")
            group = 0
            
        vertex_groups[v_idx] = group
        
    output_path = os.path.join(CURRENT_DIR, "mhr_segmentation.npy")
    np.save(output_path, vertex_groups)
    print(f"Saved segmentation to {output_path}")

if __name__ == "__main__":
    extract_segmentation()
