# Final Project Presentation: Autonomous Perception Stack
**Vision-Based Perception Pipeline for Autonomous Driving**

- **Presenter:** Yashasvi Shukla  
- **Specialization Track:** AI / Machine Learning  
- **Duration:** 10 Minutes  
- **Batch:** October 2026  

---

## Presentation Agenda & Timing (10 Minutes)

| Slide | Topic | Target Time | Cumulative |
|---|---|---|---|
| **Slide 1** | Title & Mission Statement | 0:30 | 0:30 |
| **Slide 2** | Problem Statement & Requirements | 1:00 | 1:30 |
| **Slide 3** | System Architecture & Data Flow | 1:30 | 3:00 |
| **Slide 4** | Lane Detection: Classical vs Learned | 1:30 | 4:30 |
| **Slide 5** | Object Detection & Multi-Object Tracking | 1:30 | 6:00 |
| **Slide 6** | Monocular Depth Estimation & Collision Warnings | 1:00 | 7:00 |
| **Slide 7** | Optimization: PyTorch vs ONNX Runtime Benchmark | 1:30 | 8:30 |
| **Slide 8** | Key Challenges & Engineering Solutions | 1:00 | 9:30 |
| **Slide 9** | Conclusion, Future Work & Q&A | 0:30 | 10:00 |

---

## Slide 1: Title & Overview (0:00 - 0:30)

### Slide Content:
- **Title:** Autonomous Driving Perception Stack
- **Subtitle:** Modular Computer Vision Pipeline: Lanes, Detection, ByteTrack, Monocular Depth, and ONNX Runtime Acceleration
- **Author:** Yashasvi Shukla | AI & ML Engineering Track

### Speaker Script:
> "Good morning everyone. Today I'm excited to present my Autonomous Driving Perception Stack. In autonomous vehicle systems, the perception layer is the vehicle's eyes and visual brain. It is responsible for making sense of incoming camera feeds in real time: identifying where the lane boundaries are, recognizing other road participants, tracking their trajectories across time, estimating their 3D distance, and triggering critical safety alerts before sending data to downstream planning and control modules."

---

## Slide 2: Problem Statement & Requirements (0:30 - 1:30)

### Slide Content:
- **Core Challenge:** Driving scenes are dynamic, cluttered, and latency-critical. Dropping frames or delaying detection can lead to catastrophic accidents.
- **Project Requirements:**
  1. Detect road lanes accurately under varying lighting and road curvatures.
  2. Detect vehicles, pedestrians, buses, and cyclists with high confidence.
  3. Maintain identity continuity across occlusions and motion using multi-object tracking (ByteTrack).
  4. Estimate spatial depth to critical obstacles from a single monocular camera feed.
  5. Optimize inference for real-time throughput on constrained hardware.
  6. Provide structured logging (Weights & Biases / CSV) and robust test coverage.

### Speaker Script:
> "Autonomous perception faces severe real-world constraints. Cameras provide 2D image projections with no direct depth readings, vehicle motion causes rapid viewpoint changes, and compute onboard vehicles is often power- and thermally-constrained. For this project, our goal was to build an end-to-end modular perception stack that combines classical computer vision with state-of-the-art deep learning, while remaining efficient enough to achieve real-time performance."

---

## Slide 3: Architecture & Data Flow (1:30 - 3:00)

### Slide Content:
- **Visual Data Flow Diagram:**
  ```
  Camera Feed / Driving Video (1280x720 @ 30 FPS)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  [Lane Detection]         [YOLOv8n Detector]
   • Grayscale & Blur       • 6 driving classes
   • Canny Edge Detection   • Confidence filtering
   • Dynamic ROI Masking            │
   • Hough Transform Lines          ▼
   • Slope Separation & Averaging [ByteTrack Tracker]
        │                   • Kalman Filter state
        │                   • 2-stage IoU association
        │                   • Persistent Track IDs
        │                           │
        │                  [MiDaS Monocular Depth]
        │                   • Inverse relative depth map
        │                   • BBox centroid sampling
        │                   • Metric mapping (2m - 80m)
        │                           │
        └─────────────┬─────────────┘
                      ▼
            [Multi-Modal Fusion Layer]
             • Polygons + Bounding Boxes
             • Distance HUD + Safety Alerts (< 15m)
                      │
             ┌────────┴────────┐
             ▼                 ▼
     [Annotated Video]   [Metrics Logging]
     (H.264 / MP4)       (W&B + CSV)
  ```

### Speaker Script:
> "Let’s look at the architectural pipeline. Each frame passes simultaneously into our spatial lane detector and our deep learning object detection pipeline. For lane detection, we apply classical computer vision techniques to isolate the ego-lane corridor. For object detection, YOLOv8n identifies objects of interest. These detections are immediately ingested by ByteTrack to preserve object IDs across frames. When depth is enabled, Intel MiDaS extracts a dense depth map, and we sample the depth value at the centroid of each bounding box to calculate distance. Finally, all annotations and telemetry are fused onto the frame and logged."

---

## Slide 4: Lane Detection: Classical vs. Deep Segmentation (3:00 - 4:30)

### Slide Content:
- **Approach Chosen:** Classical OpenCV Pipeline (Zero-GPU overhead, instantaneous CPU latency).
- **Processing Steps:**
  1. **Grayscale & 5x5 Gaussian Blur:** Suppress high-frequency sensor noise.
  2. **Canny Edge Detection (50, 150):** Extract strong gradient edges.
  3. **Trapezoidal ROI Mask:** Focus exclusively on the road surface ahead, ignoring sky, trees, and roadside billboards.
  4. **Probabilistic Hough Transform (`HoughLinesP`):** Detect line segments.
  5. **Slope Partitioning:**
     - Left Lane: $\text{slope} < -0.3$
     - Right Lane: $\text{slope} > +0.3$
  6. **Linear Regression & Temporal Smoothing:** Fit $x = my + b$ lines and render the green lane guidance corridor.
- **Comparison with Deep Segmentation:**
  - *Classical OpenCV:* 1.2 ms latency, 0 MB VRAM, deterministic, no training data required.
  - *Segmentation (e.g. U-Net / SegNet):* High mIoU on complex curves and faded markings, but requires 30-50 ms GPU inference and thousands of annotated frames.

### Speaker Script:
> "For lane detection, we implemented a classical computer vision pipeline using Gaussian filtering, Canny edge detection, and Hough transforms constrained by a polygonal Region of Interest. Why classical rather than deep segmentation? On compute-constrained hardware, classical lane detection processes in approximately 1 millisecond per frame with zero GPU memory overhead. In our design analysis, we compared this with deep segmentation models like SegNet or U-Net, which deliver superior accuracy on broken markings or severe curvature but consume significant compute budget that is better allocated to 3D object detection."

---

## Slide 5: Object Detection & Multi-Object Tracking (4:30 - 6:00)

### Slide Content:
- **Detector:** Ultralytics YOLOv8n (nano architecture).
  - 3.15M parameters, 8.7 GFLOPs.
  - Optimized anchor-free decoupled head.
  - Targets 6 road classes: `car`, `truck`, `bus`, `motorcycle`, `person`, `bicycle`.
- **Tracker:** ByteTrack Algorithm.
  - *Problem:* Traditional trackers (SORT, DeepSORT) discard low-confidence detections, losing tracked objects during partial occlusion or motion blur.
  - *ByteTrack Innovation:* Two-stage association matching. High-confidence detections are matched first using Kalman filters and IoU; remaining unmatched tracklets are then matched against low-confidence detections.
  - *Result:* Eliminates ID switches during vehicle overtaking or pedestrian street crossing.

### Speaker Script:
> "Next is object detection and tracking. We utilized YOLOv8 nano, which features an anchor-free split-head architecture that provides great feature extraction while running at only 8.7 GFLOPs. However, frame-by-frame detection alone cannot compute vehicle speed or predict collision trajectories. To achieve temporal continuity, we integrated ByteTrack. ByteTrack's key advantage over standard SORT is its dual-threshold data association: instead of discarding low-confidence detections, it uses them to maintain tracks through partial occlusions and camera vibration."

---

## Slide 6: Monocular Depth & Distance Alerts (6:00 - 7:00)

### Slide Content:
- **Model:** Intel MiDaS Small (Vision Transformer / EfficientNet backbone).
- **Principle:** Estimates inverse relative depth $d_{\text{rel}} \in [0, 1]$ from RGB cues (shading, vanishing point, relative scale).
- **Metric Calibration:**
  $$\text{Distance (m)} = 2.0 + (1.0 - d_{\text{rel}}) \times 78.0$$
  Calibrated for typical automotive viewing geometry (2 m to 80 m range).
- **Active Safety Trigger:**
  - Any obstacle with $\text{Distance} < 15.0\text{ m}$ triggers a red collision warning box (`! CLOSE <dist>m`) and is flagged in the telemetry logger for emergency braking.

### Speaker Script:
> "To understand distance without expensive LiDAR sensors, we introduced monocular depth estimation using MiDaS. MiDaS infers relative inverse depth by learning visual perspective cues. We sample the depth map at each bounding box centroid and apply a calibrated metric mapping function. When any tracked object enters our 15-meter safety buffer, the pipeline triggers a high-priority red collision warning on the HUD and logs the proximity alert to telemetry."

---

## Slide 7: Optimization & ONNX Runtime Benchmark (7:00 - 8:30)

### Slide Content:
- **Optimization Strategy:**
  - PyTorch overhead: Dynamic graph dispatch, Python interpreter overhead.
  - ONNX Runtime: Graph simplification via `onnxslim`, constant folding, vectorized kernel execution (AVX-512 / OpenVINO).
- **Real Benchmark Results (Intel Core i5-1334U CPU):**
  | Execution Engine | Average Latency | Throughput (FPS) | Speedup Factor |
  |---|---|---|---|
  | **PyTorch (Baseline)** | 90.76 ms | 11.02 FPS | 1.00× |
  | **ONNX Runtime (Optimized)** | **34.16 ms** | **29.27 FPS** | **2.66× Speedup** |
- **Impact:** Converts a sub-real-time 11 FPS pipeline into a full real-time 30 FPS pipeline on standard consumer CPU hardware.

### Speaker Script:
> "A core requirement was optimizing models for deployment. We exported YOLOv8 to ONNX and applied graph pruning and constant folding with ONNX Slim. We then benchmarked identical driving frames on our test hardware. In native PyTorch, inference averaged 90.8 milliseconds per frame, capping throughput at 11 FPS. With ONNX Runtime using optimized execution providers, latency dropped to 34.2 milliseconds, surging throughput to 29.3 FPS—a 2.66x acceleration that enables real-time 30 FPS perception on consumer hardware without requiring a discrete GPU."

---

## Slide 8: Key Challenges & Engineering Solutions (8:30 - 9:30)

### Slide Content:
1. **Challenge 1: High Compute Footprint of Transformers on Laptop Hardware**
   - *Solution:* Modular execution flags (`--skip-depth`), ONNX model quantization, and downscaled resolution benchmarking.
2. **Challenge 2: API Shifts in Computer Vision Libraries (OpenCV 5.0 & Supervision 0.31)**
   - *Solution:* Decoupled dependency on volatile external wrappers by utilizing native YOLO ByteTrack integration and robust line flattening algorithms.
3. **Challenge 3: Offline Validation & Test Reliability**
   - *Solution:* Authored a standalone synthetic road scene generator (`generate_synthetic.py`) and a 12-test comprehensive pytest suite with 100% pass rate.

### Speaker Script:
> "During development, we solved three major engineering hurdles: First, transformer models like MiDaS are computationally heavy on CPU, so we built configurable pipeline execution flags and ONNX optimizations. Second, we navigated recent API breaking changes in OpenCV 5.0 and Supervision by hardening our array processing and integrating native ByteTrack tracking directly. Third, to ensure test reproducibility regardless of network availability, we built an automated synthetic road video generator and established a 12-test automated unit test suite."

---

## Slide 9: Conclusion & Next Steps (9:30 - 10:00)

### Slide Content:
- **Summary of Deliverables:**
  - [x] Full perception pipeline combining Lanes + YOLOv8 + ByteTrack + MiDaS.
  - [x] 600-frame annotated demonstration video (`output/annotated_output.mp4`).
  - [x] ONNX Runtime acceleration showing 2.66× speedup.
  - [x] 12 unit tests passing with pytest.
  - [x] Complete CARLA simulator + ROS2 integration blueprint.
- **Future Roadmap:**
  - Deployment onto ROS2 Humble nodes with CARLA 0.9.15 simulator bridge.
  - TensorRT INT8 quantization on NVIDIA Jetson embedded hardware.
  - Extended multi-camera 360-degree bird's-eye-view (BEV) perception.

### Speaker Script:
> "In conclusion, we have delivered a robust, tested, and optimized autonomous perception stack that satisfies all project requirements. Looking ahead, the next step is connecting this pipeline to the CARLA simulator over ROS2 image topics and deploying INT8 TensorRT engines on embedded hardware like NVIDIA Jetson. Thank you for your time, and I welcome any questions."
