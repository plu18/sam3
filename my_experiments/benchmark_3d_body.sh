#!/bin/bash

VIDEO_PATH="my_experiments/data/bedroom.mp4"
OUTPUT_BASE="my_experiments/output_benchmark"
LOG_FILE="my_experiments/benchmark_results.txt"

# Ensure output directory exists
mkdir -p $OUTPUT_BASE

# Clear previous log
echo "Benchmark Results - $(date)" > $LOG_FILE
echo "Video: $VIDEO_PATH" >> $LOG_FILE
echo "----------------------------------------------------------------" >> $LOG_FILE
echo "| Config Name | Resize | Skip | Model Input | Vis? | FPS |" >> $LOG_FILE
echo "|---|---|---|---|---|---|" >> $LOG_FILE

run_benchmark() {
    NAME=$1
    RESIZE=$2
    SKIP=$3
    MODEL_SIZE=$4
    NO_VIS=$5
    
    echo "Running config: $NAME..."
    
    CMD="python my_experiments/process_video_3d.py \
        --video_path $VIDEO_PATH \
        --output_dir $OUTPUT_BASE/$NAME \
        --resize_width $RESIZE \
        --skip_frames $SKIP \
        --model_input_size $MODEL_SIZE"
    
    VIS_STATUS="Yes"
    if [ "$NO_VIS" = "true" ]; then
        CMD="$CMD --no_vis"
        VIS_STATUS="No"
    fi
    
    # Run and capture output
    OUTPUT=$($CMD 2>&1)
    
    # Extract FPS
    FPS=$(echo "$OUTPUT" | grep "Average FPS:" | awk '{print $NF}')
    
    if [ -z "$FPS" ]; then
        FPS="Error"
        echo "$OUTPUT" # Print output if error
    fi
    
    echo "Result: $FPS FPS"
    echo "| $NAME | $RESIZE | $SKIP | $MODEL_SIZE | $VIS_STATUS | $FPS |" >> $LOG_FILE
}

# 1. Baseline
run_benchmark "Baseline" 640 0 512 false

# 2. No Visualization
run_benchmark "No_Vis" 640 0 512 true

# 3. Smaller Video
run_benchmark "Small_Video" 480 0 512 true

# 4. Smaller Model
run_benchmark "Small_Model" 640 0 384 true

# 5. Skip Frames
run_benchmark "Skip_2" 640 2 512 true

# 6. Fast Combo
run_benchmark "Fast_Combo" 480 2 384 true

echo "Benchmark complete. Results saved to $LOG_FILE"
cat $LOG_FILE
