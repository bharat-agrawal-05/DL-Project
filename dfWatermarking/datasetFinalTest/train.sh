#!/bin/bash

# Navigate to the script directory (optional, but good practice)
# SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# cd "$SCRIPT_DIR"

echo "Starting dataset generation script (backgrounded with nohup)..." > completeLog.txt # Overwrite log initially
echo "Command: python generate_dataset.py --dataset_dir \"./TokenPatternsDataset\" --images_per_subdir 50 --embedding_steps 5 50 99 --inference_steps 100 --global_seed 2024 --embedding_strength 0.04 --embedding_num_tokens_dim 8 --overwrite" >> completeLog.txt
echo "Timestamp: $(date)" >> completeLog.txt
echo "----------------------------------------------------" >> completeLog.txt

# Execute the Python script
# Output is redirected to completeLog.txt within the Python command itself
# so nohup.out will likely only contain nohup's own messages or be empty if python script handles all output.
python generate_dataset.py \
    --dataset_dir "./TokenPatternsDataset" \
    --images_per_subdir 50 \
    --embedding_steps 5 50 99 \
    --inference_steps 100 \
    --global_seed 2024 \
    --embedding_strength 0.04 \
    --embedding_num_tokens_dim 8 \
    --overwrite >> completeLog.txt 2>&1 # Append stdout and stderr to completeLog.txt

EXIT_CODE=$?

echo "----------------------------------------------------" >> completeLog.txt
echo "Python script part finished with exit code: $EXIT_CODE" >> completeLog.txt
echo "Timestamp: $(date)" >> completeLog.txt

if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully." >> completeLog.txt
else
    echo "Python script failed. Check completeLog.txt for details." >> completeLog.txt
fi

# The main script train.sh will exit here, but the python process (if nohup'd correctly from outside) continues.
exit $EXIT_CODE