# Intern Reflection & Engineering Ownership Report
**Project:** Autonomous Driving Perception Stack  
**Intern / Author:** Yashasvi Shukla  
**Track:** AI / Machine Learning  
**Date:** October 2026  

---

## 1. Project Ownership & Areas of Responsibility

As the owner of the Autonomous Driving Perception Stack project, I was responsible for designing, implementing, benchmarking, and testing the entire vision perception pipeline. Specifically, my ownership covered:

### 1.1 Architecture & Pipeline Design
- **System Modularity:** Structured the project into distinct, loosely-coupled modules (`models/lane`, `models/detection`, `models/depth`, `pipeline`, `optimization`, `experiments`, and `tests`) so that individual components can be developed, tested, and upgraded independently.
- **Fusion Layer:** Implemented multi-modal sensor fusion logic that combines 2D bounding boxes, persistent object identities, road lane polygons, and dense depth matrices into a single annotated HUD feed.
- **Graceful Fallbacks:** Designed the architecture to function with or without Weights & Biases (local CSV fallback) and with or without GPU acceleration (CPU-optimized ONNX runtime execution).

### 1.2 Module Implementation & Engineering Decisions
1. **Classical Lane Detection (`opencv_lane.py`):**
   - Implemented Gaussian blur, Canny edge detection, trapezoidal ROI filtering, and probabilistic Hough transforms.
   - Handled slope partitioning (negative for left lane, positive for right lane) and linear extrapolation to draw an intuitive ego-lane corridor.
   - Re-engineered line parsing to handle numpy dimensionality changes in OpenCV 5.0.
2. **Object Detection & Tracking (`yolo_tracker.py`):**
   - Configured Ultralytics YOLOv8n to prioritize 6 critical road-safety classes: `car`, `truck`, `bus`, `motorcycle`, `person`, and `bicycle`.
   - Integrated ByteTrack for multi-object tracking, maintaining track persistence across frames and occlusions.
   - Designed centroid sampling logic to query the depth map and assign real-time metric distance estimates to each tracked obstacle.
   - Built safety threshold triggers to flash danger warnings whenever an obstacle is closer than 15 meters.
3. **Monocular Depth Estimation (`midas_depth.py`):**
   - Integrated Intel MiDaS Small to provide dense relative depth without requiring expensive LiDAR hardware.
   - Formulated a metric calibration curve ($2.0\text{ m} - 80.0\text{ m}$) mapping inverse relative depth to physical distance.
4. **ONNX Export & Benchmarking Engine (`export_onnx.py`):**
   - Exported YOLOv8 weights to ONNX format, streamlined graphs with `onnxslim`, and developed an automated benchmark harness to measure per-frame latency and FPS.

---

## 2. Key Technical Learnings

### 2.1 Classical Computer Vision vs. Deep Learning Trade-offs
A major learning from this project was understanding when *not* to use deep learning. While semantic segmentation models (such as SegNet or U-Net) can learn lane markings under extreme weather, classical edge and line detection runs in ~1 ms on a CPU with zero memory training requirements. Balancing classical techniques for deterministic geometry (lanes) with deep neural networks for rich semantic understanding (vehicles and pedestrians) yielded an optimal balance of throughput and accuracy.

### 2.2 Tracking by Detection & The Power of ByteTrack
Standard tracking algorithms (like SORT) drop bounding boxes below a high confidence threshold (e.g. 0.5), which causes tracked vehicles to lose their identity during sudden shadows, glare, or partial occlusion. Learning how ByteTrack preserves and associates second-stage low-confidence detections was eye-opening—it significantly stabilized tracking IDs throughout the video sequence.

### 2.3 Hardware Constraints & Production Optimization
Developing on a system with an Intel Core i5 CPU without a dedicated NVIDIA CUDA GPU made efficiency an immediate priority rather than an afterthought. Achieving **29.27 FPS (34.16 ms latency)** using ONNX Runtime—compared to 11.02 FPS in native PyTorch—demonstrated that model compilation, graph optimization, and runtime kernel selection are just as important as neural network architecture design in real-world autonomous systems.

### 2.4 Defensive Programming & Test-Driven Development
I wrote 12 automated unit tests across lane detection and object tracking modules. Setting up unit tests early exposed edge cases such as empty frames, non-standard resolutions, and API compatibility quirks between library versions.

---

## 3. Challenges Encountered & Resolutions

| Challenge | Root Cause | Engineering Resolution |
|---|---|---|
| **API Shifts in Library Releases** | `supervision 0.31` removed `sv.ByteTrack`, and `OpenCV 5.0` changed `HoughLinesP` return dimensions. | Refactored `yolo_tracker.py` to leverage YOLO's native ByteTrack engine (`tracker="bytetrack.yaml"`) and added robust `.flatten()` unwrapping to ensure version-agnostic array parsing. |
| **Download & Network Fragility** | External sample dashcam videos failed to download due to SSL issues and broken YouTube scrapers. | Developed a self-contained synthetic driving video generator (`generate_synthetic.py`) using OpenCV drawing primitives, generating a reproducible 600-frame test suite offline. |
| **Compute Bottleneck on CPU** | Monocular depth estimation via transformers is computationally expensive for high frame-rate processing. | Implemented a `--skip-depth` CLI flag for ultra-fast tracking, alongside downscaled inference passes and ONNX Runtime CPU execution provider optimizations. |

---

## 4. Personal Reflection & Next Steps

This project reinforced my passion for autonomous systems engineering. Managing the full lifecycle—from data generation and neural network inference to tracking filters, optimization benchmarks, and documentation—gave me end-to-end perspective on the autonomous driving software stack.

If granted additional time and access to dedicated GPU hardware, my next steps would be:
1. **CARLA + ROS2 Bridge Execution:** Deploy the pipeline inside a live CARLA simulator environment using the ROS2 Humble bridge outlined in `docs/carla_ros2_extension.md`.
2. **TensorRT INT8 Quantization:** Generate INT8 calibration caches on NVIDIA Jetson or RTX GPUs for sub-5ms inference.
3. **Temporal Filtering:** Implement an Extended Kalman Filter (EKF) to fuse lane geometry across frames for smoother trajectory forecasting.
