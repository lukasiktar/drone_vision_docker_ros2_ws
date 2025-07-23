# Docker Implementation of the Drone Vision Control (ROS 2)

This repository contains a Dockerized implementation of a drone vision control system using ROS 2, based on the algorithm described in the research paper:

> **Autonomous Quadcopter Navigation for Search and Rescue Missions Using Computer Vision and Convolutional Neural Networks**  
> (with modifications tailored to the Hexoon drone)

## Overview

This system enables autonomous navigation for the Hexoon drone using computer vision and deep learning techniques. The implementation is designed for ROS 2 and runs inside a Docker container for easy deployment and reproducibility.

It works **in conjunction** with the [`robocamp_jetson_server`](https://github.com/CRTA-Lab/robocamp_jetson_server) package, which provides the backend control interface for the Hexoon drone.

## Features

- ROS 2-based drone vision control
- Deep learning-based object detection and navigation
- Dockerized environment for plug-and-play usage
- Compatible with RealSense depth cameras

## System Requirements

- Hexoon drone
- Jetson device (e.g., Xavier NX, Orin)
- RealSense depth camera (e.g., D435i)
- Docker & NVIDIA Container Toolkit
- [`robocamp_jetson_server`](https://github.com/CRTA-Lab/robocamp_jetson_server)

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/lukasiktar/drone-vision-control-docker.git
   cd drone-vision-control-docker
