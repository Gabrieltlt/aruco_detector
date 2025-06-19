## Overview
This is a ROS 2 Humble package for aruco detection.

### Main Package
- arm_behavior;

## Instalation
Run the follow commands:
```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/main_ws/src
cd ~/main_ws/src
https://github.com/Gabrieltlt/aruco_detector.git
cd ~/main_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build
```

## Usage
To start aruco recognition, run:
```bash
ros2 launch aruco_detector aruco_recognition.launch.py
```