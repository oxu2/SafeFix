#!/bin/bash

# Usage:
# CUDA_VISIBLE_DEVICES=2 nohup bash run.sh your_script.py --args > ./logs/run.log 2>&1 &

# Record start time in full format
start_time=$(date +"%F %T")
start_seconds=$(date +%s)

echo "start time: $start_time"

# Get Python script and its arguments
SCRIPT="$1"
shift
ARGS=("$@")

# Run the Python script
python "$SCRIPT" "${ARGS[@]}"

# Record end time
end_time=$(date +"%F %T")
end_seconds=$(date +%s)

# Calculate duration in HH:MM:SS
duration_seconds=$((end_seconds - start_seconds))
duration=$(printf '%02d:%02d:%02d' $((duration_seconds/3600)) $(( (duration_seconds%3600)/60 )) $((duration_seconds%60)))

echo "Start Time: $start_time"
echo "End Time:   $end_time"
echo "Duration:   $duration"