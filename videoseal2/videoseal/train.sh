#!/bin/bash

echo "Starting generateDATASET.py in the background..."
echo "Standard output and error will be discarded."

# Run the torchrun command with nohup
# stdout and stderr are redirected to /dev/null to discard them
nohup torchrun --nproc_per_node=1 generateDATASET.py > /dev/null 2>&1 &

# Get the PID of the last backgrounded process
PID=$!

# Print the PID to the terminal
echo "generateDATASET.py started with PID: $PID"
echo "This process will continue running after you close the terminal."
echo "To check its status (e.g., if it's still running): ps -p $PID"
echo "To stop the process: kill $PID"

exit 0