
import sys
import os
import torch
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

def inspect_mhr():
    checkpoint_path = "checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint_path}...")
    model, _ = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    
    print("Model loaded.")
    
    # Inspect MHR structure
    if hasattr(model, 'head_pose') and hasattr(model.head_pose, 'mhr'):
        mhr = model.head_pose.mhr
        print("MHR object found.")
        print(f"Type: {type(mhr)}")
        print(f"Dir: {dir(mhr)}")
        
        if hasattr(mhr, 'character_torch'):
            char = mhr.character_torch
            print("\nCharacter Torch found.")
            print(f"Dir: {dir(char)}")
            
            if hasattr(char, 'mesh'):
                mesh = char.mesh
                print("\nMesh found.")
                print(f"Dir: {dir(mesh)}")
                if hasattr(mesh, 'rest_vertices'):
                    print(f"Rest vertices shape: {mesh.rest_vertices.shape}")
                if hasattr(mesh, 'faces'):
                    print(f"Faces shape: {mesh.faces.shape}")
                    
            if hasattr(char, 'linear_blend_skinning'):
                lbs = char.linear_blend_skinning
                print("\nLinear Blend Skinning found.")
                print(f"Dir: {dir(lbs)}")
                if hasattr(lbs, 'skin_weights_flattened'):
                    print(f"Skin weights flattened shape: {lbs.skin_weights_flattened.shape}")
                if hasattr(lbs, 'skin_indices_flattened'):
                    print(f"Skin indices flattened shape: {lbs.skin_indices_flattened.shape}")
                if hasattr(lbs, 'vert_indices_flattened'):
                    print(f"Vert indices flattened shape: {lbs.vert_indices_flattened.shape}")
                if hasattr(lbs, 'inverse_bind_pose'):
                    print(f"Inverse bind pose shape: {lbs.inverse_bind_pose.shape}")

            if hasattr(char, 'skeleton'):
                skel = char.skeleton
                print("\nSkeleton found.")
                print(f"Dir: {dir(skel)}")
                if hasattr(skel, 'joint_names'):
                    print(f"Joint names: {skel.joint_names}")
                elif hasattr(skel, 'names'):
                    print(f"Names: {skel.names}")
                
                # Try to infer joint count
                if hasattr(skel, 'joint_parents'):
                    print(f"Joint parents shape: {skel.joint_parents.shape}")

    else:
        print("MHR object NOT found in model.head_pose")

if __name__ == "__main__":
    inspect_mhr()
