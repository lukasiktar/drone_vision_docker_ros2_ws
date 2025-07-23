#!/bin/bash
# This script builds the workspace, sets up environment variables, and runs the realsense_control node

# Step 1: Build the workspace
colcon build

# Step 2: Set the GPU device
export CUDA_VISIBLE_DEVICES=0

# Step 3: Source the ROS 2 workspace
source install/setup.bash

# Step 4: Set the ROS Domain ID to avoid conflicts
export ROS_DOMAIN_ID=3

# Step 5: Run the realsense_control node
ros2 run realsense_control realsense_control
