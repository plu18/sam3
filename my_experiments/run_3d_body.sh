#!/bin/bash

# Run SAM 3D Body on all videos in the sports_videos directory
python my_experiments/process_video_3d.py \
  --video_path my_experiments/data/sports_videos/football_1.mp4 \
  --output_dir my_experiments/output_3d_body \
  --checkpoint_path checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
