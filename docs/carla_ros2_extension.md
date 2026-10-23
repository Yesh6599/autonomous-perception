# CARLA + ROS2 Extension Guide

> This guide explains how to extend the pipeline to run live inside the CARLA driving simulator via ROS2.
> **Requires: NVIDIA GPU, Linux (Ubuntu 20.04/22.04), CARLA 0.9.15, ROS2 Humble**

---

## Prerequisites

```bash
# 1. Install CARLA 0.9.15
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz

# 2. Install ROS2 Humble
# Follow: https://docs.ros.org/en/humble/Installation.html

# 3. Install CARLA ROS2 Bridge
pip install carla==0.9.15
```

---

## CARLA Camera Bridge

```python
# pipeline/carla_bridge.py (pseudocode)
import carla
import rclpy
from sensor_msgs.msg import Image

client = carla.Client("localhost", 2000)
world  = client.get_world()

# Spawn ego vehicle
blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
vehicle   = world.spawn_actor(blueprint, spawn_point)

# Attach RGB camera
cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1280")
cam_bp.set_attribute("image_size_y", "720")
cam_bp.set_attribute("fov", "90")
camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=vehicle)

# Publish to ROS2 topic
def on_image(image):
    ros_image = carla_image_to_ros(image)
    publisher.publish(ros_image)

camera.listen(on_image)
```

---

## ROS2 Perception Node

```python
# pipeline/perception_node.py (pseudocode)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.pipeline = PerceptionPipeline()
        self.sub  = self.create_subscription(Image, "/camera/image_raw", self.callback, 10)
        self.pub  = self.create_publisher(Image, "/perception/annotated", 10)
        self.brake_pub = self.create_publisher(Float32, "/vehicle/brake", 10)

    def callback(self, msg):
        frame     = ros_image_to_cv2(msg)
        annotated = self.pipeline.process_frame(frame)
        self.pub.publish(cv2_to_ros_image(annotated))

        # Send brake if obstacle within 15m
        closest = self.pipeline.get_closest_distance()
        if closest and closest < 15.0:
            self.brake_pub.publish(Float32(data=1.0))
```

---

## Running

```bash
# Terminal 1: Start CARLA
./CarlaUE4.sh

# Terminal 2: Start ROS2 bridge
ros2 run autonomous_perception carla_bridge

# Terminal 3: Start perception node
ros2 run autonomous_perception perception_node
```
