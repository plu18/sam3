import pickle
import numpy as np
import os
import argparse

# MHR70 Joint Names
MHR_NAMES = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-hip", "right-hip", "left-knee", "right-knee",
    "left-ankle", "right-ankle", "left-big-toe-tip", "left-small-toe-tip", "left-heel",
    "right-big-toe-tip", "right-small-toe-tip", "right-heel",
    "right-thumb-tip", "right-thumb-first-joint", "right-thumb-second-joint", "right-thumb-third-joint",
    "right-index-tip", "right-index-first-joint", "right-index-second-joint", "right-index-third-joint",
    "right-middle-tip", "right-middle-first-joint", "right-middle-second-joint", "right-middle-third-joint",
    "right-ring-tip", "right-ring-first-joint", "right-ring-second-joint", "right-ring-third-joint",
    "right-pinky-tip", "right-pinky-first-joint", "right-pinky-second-joint", "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip", "left-thumb-first-joint", "left-thumb-second-joint", "left-thumb-third-joint",
    "left-index-tip", "left-index-first-joint", "left-index-second-joint", "left-index-third-joint",
    "left-middle-tip", "left-middle-first-joint", "left-middle-second-joint", "left-middle-third-joint",
    "left-ring-tip", "left-ring-first-joint", "left-ring-second-joint", "left-ring-third-joint",
    "left-pinky-tip", "left-pinky-first-joint", "left-pinky-second-joint", "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon", "right-olecranon", "left-cubital-fossa", "right-cubital-fossa",
    "left-acromion", "right-acromion", "neck"
]

def write_trc(filename, trajectory, fps=30.0):
    """
    Write a TRC file for OpenSim.
    trajectory: list of dicts (or None) with 'joints' key.
    """
    num_frames = len(trajectory)
    num_markers = len(MHR_NAMES)
    
    # Find start frame
    start_frame = 1
    
    with open(filename, 'w') as f:
        # Header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(filename)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tm\t{fps}\t{start_frame}\t{num_frames}\n")
        
        # Marker Names
        f.write("Frame#\tTime\t")
        for name in MHR_NAMES:
            f.write(f"{name}\t\t\t")
        f.write("\n")
        
        f.write("\t\t")
        for i in range(num_markers):
            f.write(f"X{i+1}\tY{i+1}\tZ{i+1}\t")
        f.write("\n")
        
        f.write("\n") # Empty line usually required
        
        # Data
        for i, item in enumerate(trajectory):
            frame_num = start_frame + i
            time = i / fps
            
            f.write(f"{frame_num}\t{time:.4f}\t")
            
            if item is None or 'data' not in item or 'joints' not in item['data']:
                # Write NaNs or zeros for missing data
                # OpenSim usually prefers empty or specific missing value? 
                # Usually 0.0 or NaN. Let's use 0.0 for now, or interpolate.
                # Better to skip or use last valid? TRC expects rows for all frames.
                for _ in range(num_markers):
                    f.write("0.0000\t0.0000\t0.0000\t")
            else:
                joints = item['data']['joints'] # Shape (70, 3)
                
                # Coordinate Transformation
                # SAM3 (CV): X-right, Y-down, Z-forward
                # OpenSim: X-forward, Y-up, Z-right
                # Transformation:
                # OS_X = SAM3_Z
                # OS_Y = -SAM3_Y
                # OS_Z = SAM3_X
                
                for j in range(num_markers):
                    x_sam = joints[j, 0]
                    y_sam = joints[j, 1]
                    z_sam = joints[j, 2]
                    
                    # Apply transformation
                    x_os = z_sam
                    y_os = -y_sam
                    z_os = x_sam
                    
                    f.write(f"{x_os:.4f}\t{y_os:.4f}\t{z_os:.4f}\t")
            
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Convert SAM3 PKL to OpenSim TRC")
    parser.add_argument("--input", required=True, help="Input PKL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for TRC files")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Loading {args.input}...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
        
    # Iterate over tracks
    for track_id, trajectory in data.items():
        print(f"Processing Track {track_id}...")
        output_filename = os.path.join(args.output_dir, f"track_{track_id}.trc")
        write_trc(output_filename, trajectory, fps=args.fps)
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    main()
