# Autonomous Driving Perception Stack

> A full computer vision pipeline for detecting lanes, vehicles, and pedestrians in driving footage — with multi-object tracking and monocular depth estimation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/detection-YOLOv8n-orange)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/tracking-ByteTrack-green)](https://github.com/roboflow/supervision)
[![MiDaS](https://img.shields.io/badge/depth-MiDaS_Small-purple)](https://github.com/isl-org/MiDaS)
[![W&B](https://img.shields.io/badge/logging-W%26B-yellow)](https://wandb.ai)

---

## 📸 What It Does

Given a driving video, the pipeline produces a fully annotated output with:

| Module | Technology | Output |
|---|---|---|
| 🛣️ Lane Detection | OpenCV (Canny + Hough) | Green lane overlay + filled region |
| 🚗 Object Detection | YOLOv8n (pretrained) | Bounding boxes per class |
| 🎯 Multi-Object Tracking | ByteTrack (via supervision) | Persistent track IDs across frames |
| 📏 Depth Estimation | MiDaS Small (CPU) | Distance label per object + depth heatmap PiP |
| 📊 Experiment Logging | Weights & Biases + CSV | Per-frame FPS, counts, distances |

---

## 🖥️ System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10/11, Linux, macOS |
| Python | 3.10 or newer |
| RAM | 8 GB (16 GB recommended) |
| GPU | Optional — CPU is supported (Intel Iris Xe tested) |
| Storage | ~2 GB for models + video |

> **Note:** TensorRT optimisation requires an NVIDIA GPU with CUDA. On CPU-only systems, ONNX Runtime is used instead.

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/autonomous-perception.git
cd autonomous-perception

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download a Sample Video

```bash
python data/download_sample.py
```

Or place any dashcam `.mp4` file in `data/` and reference it below.

### 3. Run the Pipeline

**Fast mode (no depth — good for CPU):**
```bash
python pipeline/run_pipeline.py --input data/sample_driving.mp4 --skip-depth --show
```

**Full mode (with depth estimation):**
```bash
python pipeline/run_pipeline.py --input data/sample_driving.mp4 --show
```

**Process silently and save output:**
```bash
python pipeline/run_pipeline.py --input data/sample_driving.mp4 --output output/
```

Output video is saved to `output/annotated_output.mp4`.

---

## 📁 Project Structure

```
autonomous-perception/
├── models/
│   ├── lane/
│   │   └── opencv_lane.py        ← Classical lane detection
│   ├── detection/
│   │   └── yolo_tracker.py       ← YOLOv8n + ByteTrack
│   └── depth/
│       └── midas_depth.py        ← MiDaS monocular depth
├── pipeline/
│   └── run_pipeline.py           ← Main entry point
├── experiments/
│   └── wandb_logger.py           ← W&B + CSV logging
├── optimization/
│   └── export_onnx.py            ← ONNX export + benchmark
├── tests/
│   ├── test_lane.py
│   └── test_yolo.py
├── data/
│   └── download_sample.py        ← Sample video downloader
├── output/                       ← Generated outputs (gitignored)
├── requirements.txt
└── README.md
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_lane.py::test_detect_lanes_returns_tuple        PASSED
tests/test_lane.py::test_detect_lanes_annotated_frame_shape PASSED
tests/test_lane.py::test_detect_lanes_finds_lines_on_synthetic PASSED
tests/test_lane.py::test_detect_lanes_black_frame          PASSED
tests/test_yolo.py::test_tracker_loads                     PASSED
tests/test_yolo.py::test_process_frame_returns_tuple       PASSED
...
```

---

## 📊 ONNX Benchmarking

Export the YOLO model to ONNX and compare inference speed:

```bash
python optimization/export_onnx.py --input data/sample_driving.mp4 --frames 40
```

Actual measured results on 13th Gen Intel Core i5-1334U (CPU only):

| Execution Engine | Latency (ms/frame) | Throughput (FPS) | Speedup Factor |
|---|---|---|---|
| **PyTorch (Baseline)** | 90.76 ms | 11.02 FPS | 1.00× |
| **ONNX Runtime (Optimized)** | **34.16 ms** | **29.27 FPS** | **2.66×** |

> Model weights exported to `yolov8n.onnx` (12.5 MB) using opset 18 and graph slimming via `onnxslim`.


---

## 📈 Weights & Biases Logging

Set your API key and it logs automatically:

```powershell
# Windows PowerShell
$env:WANDB_API_KEY = "your_key_here"
python pipeline/run_pipeline.py --input data/sample_driving.mp4
```

W&B logs:
- Per-frame FPS and latency
- Detection counts per class
- Closest object distance
- Output video as W&B artifact

A local `output/metrics.csv` is always saved regardless of W&B.

---

## 🏗️ Architecture

```
Input Video / Camera
        │
        ▼
┌───────────────────────────────────────────────┐
│                Perception Pipeline            │
│                                               │
│  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Lane Det.    │  │ YOLOv8n + ByteTrack  │  │
│  │ (OpenCV)     │  │ (detect + track)     │  │
│  └──────────────┘  └──────────┬───────────┘  │
│                               │              │
│                    ┌──────────▼───────────┐  │
│                    │  MiDaS Depth Est.    │  │
│                    │  (distance per bbox) │  │
│                    └──────────┬───────────┘  │
│                               │              │
│          ┌────────────────────▼──────────┐   │
│          │   Fusion + Annotation Layer   │   │
│          └────────────────────┬──────────┘   │
└───────────────────────────────┼──────────────┘
                                │
               ┌────────────────▼──────────┐
               │  W&B Logger + CSV Output  │
               └───────────────────────────┘
                                │
                    Annotated Output Video
```

---

## 🚀 CARLA + ROS2 (Future Extension)

> ⚙️ *Requires NVIDIA GPU + Linux + CARLA 0.9.15 + ROS2 Humble*

This pipeline is designed to be ROS2-compatible. The `PerceptionPipeline` class can be wrapped in a ROS2 node that:
1. Subscribes to `/camera/image_raw` from a CARLA virtual camera
2. Runs the full pipeline on each frame
3. Publishes annotated images + detected objects
4. Sends brake commands when obstacles are within 15m

See `docs/carla_ros2_extension.md` for setup instructions.

---

## 📋 Deliverables Checklist

- [x] **Working Demo Video:** 600-frame annotated autonomous driving video generated (`output/annotated_output.mp4`)
- [x] **GitHub Repository:** Clean modular architecture, `.gitignore`, and detailed setup instructions
- [x] **Architecture Diagram:** Comprehensive visualization saved at [`docs/architecture.jpg`](docs/architecture.jpg)
- [x] **Test Suite & Performance Results:** 12/12 pytest unit tests passing; ONNX benchmark confirms **2.66× speedup (29.27 FPS)**
- [x] **10-Minute Final Presentation:** Slide deck outline and complete speaker notes at [`docs/presentation.md`](docs/presentation.md)
- [x] **Intern Reflection Report:** Engineering ownership, learnings, and trade-offs at [`docs/intern_reflection.md`](docs/intern_reflection.md)
- [x] **CARLA & ROS2 Integration Blueprint:** Full node architecture guide at [`docs/carla_ros2_extension.md`](docs/carla_ros2_extension.md)


---

## 🛠️ Tech Stack

| Component | Library / Tool |
|---|---|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Multi-Object Tracking | [supervision ByteTrack](https://github.com/roboflow/supervision) |
| Lane Detection | OpenCV 4.x |
| Depth Estimation | [Intel MiDaS](https://github.com/isl-org/MiDaS) |
| Model Export | ONNX + ONNX Runtime |
| Experiment Tracking | [Weights & Biases](https://wandb.ai) |
| Testing | pytest |

---

## 👤 Author

**Yashasvi Shukla**  
AI/ML Internship Project — Batch Oct 2026  
Deadline: 20 November 2026
