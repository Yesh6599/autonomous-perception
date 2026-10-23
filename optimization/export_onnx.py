"""
ONNX Export Script
===================
Exports YOLOv8n to ONNX format and benchmarks:
  - PyTorch inference
  - ONNX Runtime inference

Usage:
  python optimization/export_onnx.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

def export_yolo_to_onnx(model_path: str = "yolov8n.pt") -> str:
    """Export YOLOv8n to ONNX and return the output path."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    out = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"[ONNX] Exported to: {out}")
    return str(out)


def benchmark(video_path: str, n_frames: int = 100) -> dict:
    """
    Benchmark PyTorch vs ONNX on real frames.

    Returns dict with avg latency and FPS for each backend.
    """
    from ultralytics import YOLO
    import onnxruntime as ort

    # ── Load models ───────────────────────────────────────────────────────
    pt_model   = YOLO("yolov8n.pt")
    onnx_path  = export_yolo_to_onnx("yolov8n.pt")
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # ── Grab sample frames ────────────────────────────────────────────────
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.resize(frame, (640, 384)))
    cap.release()

    print(f"\n[Benchmark] Testing on {len(frames)} frames …")

    # ── PyTorch timing ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for f in frames:
        pt_model(f, verbose=False)
    pt_time = time.perf_counter() - t0
    pt_fps  = len(frames) / pt_time
    pt_lat  = pt_time / len(frames) * 1000  # ms

    # ── ONNX Runtime timing ───────────────────────────────────────────────
    input_name = ort_session.get_inputs()[0].name
    t0 = time.perf_counter()
    for f in frames:
        blob = cv2.dnn.blobFromImage(f, 1 / 255.0, (640, 384), swapRB=True)
        ort_session.run(None, {input_name: blob})
    ort_time = time.perf_counter() - t0
    ort_fps  = len(frames) / ort_time
    ort_lat  = ort_time / len(frames) * 1000

    results = {
        "pytorch_avg_latency_ms": round(pt_lat, 2),
        "pytorch_fps":            round(pt_fps, 2),
        "onnx_avg_latency_ms":    round(ort_lat, 2),
        "onnx_fps":               round(ort_fps, 2),
        "speedup":                round(pt_lat / ort_lat, 2),
    }

    print("\n" + "=" * 40)
    print("  BENCHMARK RESULTS")
    print("=" * 40)
    print(f"  PyTorch  : {pt_fps:.1f} FPS  ({pt_lat:.1f} ms/frame)")
    print(f"  ONNX     : {ort_fps:.1f} FPS  ({ort_lat:.1f} ms/frame)")
    print(f"  Speedup  : {results['speedup']}×")
    print("=" * 40)

    # Save to file
    out_path = ROOT / "output" / "benchmark_results.txt"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"\n[Benchmark] Results saved → {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Sample video for benchmark")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    benchmark(args.input, args.frames)
