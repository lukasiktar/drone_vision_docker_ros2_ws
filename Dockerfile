# Use ROS as the base image
FROM dustynv/ros:humble-ros-base-l4t-r36.2.0

# Install system dependencies and ROS dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
    gnupg2 \
    lsb-release \
    git \
    python3-rosdep \
    cuda-command-line-tools-12-2 \
    cuda-nvcc-12-2 \
    cuda-libraries-dev-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link if necessary
RUN ln -s /usr/lib/aarch64-linux-gnu/libmpi.so /usr/lib/aarch64-linux-gnu/libmpi.so.20 || true

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.2 support
RUN wget https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl -O torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl \
    && pip3 install 'Cython<3' \
    && pip3 install numpy torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl \
    && pip install --upgrade typing_extensions \
    && pip install numpy==1.24.2

# Install TorchVision from source
RUN git clone --branch v0.17.1 https://github.com/pytorch/vision torchvision \
    && cd torchvision \
    && export BUILD_VERSION=0.17.1 \
    && python3 setup.py install --user \
    && cd ..

# Install libraries
RUN pip3 install ultralytics supervision lap face_recognition

# Install ROS2 joy
RUN apt update && apt install -y \
    ros-humble-joy \
    ros-humble-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*  

# Set up working directory for your project
WORKDIR /app

# Copy your ROS and NN inference files into the container
COPY . .

#Make run.sh executable
RUN chmod +x run.sh

# Define an entry point for your ROS-based NN inference
CMD ["/bin/bash"]

# Test CUDA installation
RUN nvcc --version && nvidia-smi

