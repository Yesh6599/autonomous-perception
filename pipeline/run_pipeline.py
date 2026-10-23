"""
Autonomous Driving Perception Pipeline
========================================
Main pipeline that fuses:
  - Lane Detection (OpenCV)
  - Object Detection + Tracking (YOLOv8n + ByteTrack)
  - Depth Estimation (MiDaS Small)

Processes either:
  - A video file (--input path/to/video.mp4)
  - A webcam   (--input 0)

Usage:
  python pipeline/run_pipeline.py --input data/sample.mp4
  python pipeline/run_pipeline.py --input data/sample.mp4 --skip-depth
  python pipeline/run_pipeline.py --input data/sample.mp4 --show
"""

import sys
import time
import argparse
from pathlib import Path

# ── Allow imports from project root ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from models.lane.opencv_lane       import detect_lanes
from models.detection.yolo_tracker import ObjectTracker
from models.depth.midas_depth      import DepthEstimator, blend_depth_overlay
from experiments.wandb_logger      import PerceptionLogger


# ── Pipeline class ─────────────────────────────────────────────────────────

class PerceptionPipeline:
    """
    End-to-end autonomous driving perception pipeline.
    """

    def __init__(
        self,
        skip_depth:   bool = False,
        yolo_conf:    float = 0.35,
        depth_alpha:  float = 0.25,
        use_wandb:    bool = True,
        output_dir:   str  = "output",
    ):
        """
        Args:
            skip_depth:  Skip MiDaS depth (faster — no depth labels)
            yolo_conf:   YOLO confidence threshold
            depth_alpha: Opacity of depth overlay on frame
            use_wandb:   Enable W&B logging
            output_dir:  Directory to save output video + CSV
        """
        self.skip_depth  = skip_depth
        self.depth_alpha = depth_alpha
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n══════════════════════════════════════════")
        print("  Autonomous Perception Pipeline")
        print("══════════════════════════════════════════")

        self.tracker = ObjectTracker(conf=yolo_conf)

        if not skip_depth:
            self.depth_estimator = DepthEstimator("MiDaS_small")
        else:
            self.depth_estimator = None
            print("[Pipeline] Depth estimation SKIPPED")

        self.logger = PerceptionLogger(
            output_dir=output_dir,
            use_wandb=use_wandb,
        )

        print("══════════════════════════════════════════\n")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full pipeline on a single frame.

        Args:
            frame: BGR image

        Returns:
            Fully annotated BGR frame
        """
        t_start = time.perf_counter()

        # ── 1. Depth Estimation ───────────────────────────────────────────
        depth_map     = None
        depth_colored = None
        if self.depth_estimator is not None:
            depth_map, depth_colored = self.depth_estimator.estimate(frame)

        # ── 2. Object Detection + Tracking ────────────────────────────────
        det_frame, detections = self.tracker.process_frame(frame, depth_map)

        # ── 3. Lane Detection ─────────────────────────────────────────────
        annotated, lane_lines = detect_lanes(det_frame)
        lanes_ok = len(lane_lines) >= 2

        # ── 4. Depth Overlay (small picture-in-picture) ───────────────────
        if depth_colored is not None:
            annotated = _draw_depth_pip(annotated, depth_colored)

        # ── 5. Global HUD ─────────────────────────────────────────────────
        annotated = _draw_global_hud(annotated, detections, lanes_ok)

        # ── 6. Log metrics ────────────────────────────────────────────────
        frame_time = time.perf_counter() - t_start
        self.logger.log_frame(detections, lanes_ok, frame_time)

        return annotated

    def run(self, source: str, show: bool = False, max_frames: int | None = None) -> str:
        """
        Process an entire video source and save annotated output.

        Args:
            source:     Path to video file or camera index string (e.g. "0")
            show:       Show live preview window
            max_frames: Limit frames processed (None = all)

        Returns:
            Path to the saved output video
        """
        # ── Open source ───────────────────────────────────────────────────
        cap_src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")

        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Pipeline] Source  : {source}")
        print(f"[Pipeline] Size    : {width}×{height}")
        print(f"[Pipeline] FPS     : {fps_src:.1f}")
        print(f"[Pipeline] Frames  : {total if total > 0 else 'unknown'}")

        # ── Output writer ─────────────────────────────────────────────────
        out_path = self.output_dir / "annotated_output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps_src, (width, height))

        # ── Main loop ─────────────────────────────────────────────────────
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                annotated = self.process_frame(frame)
                writer.write(annotated)

                if show:
                    cv2.imshow("Autonomous Perception Pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Pipeline] Stopped by user (q)")
                        break

                frame_idx += 1

                # Progress every 30 frames
                if frame_idx % 30 == 0:
                    pct = f"{frame_idx/total*100:.1f}%" if total > 0 else f"{frame_idx} frames"
                    print(f"[Pipeline] Progress: {pct}", end="\r")

        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

        print(f"\n[Pipeline] Output saved → {out_path}")
        self.logger.log_video(str(out_path))
        self.logger.close()
        return str(out_path)


# ── Helper draw functions ──────────────────────────────────────────────────

def _draw_depth_pip(frame: np.ndarray, depth_colored: np.ndarray) -> np.ndarray:
    """Draw depth map as a small picture-in-picture in the bottom-right."""
    h, w = frame.shape[:2]
    pip_w = w // 4
    pip_h = h // 4
    depth_small = cv2.resize(depth_colored, (pip_w, pip_h))

    # Border
    cv2.rectangle(depth_small, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 1)
    cv2.putText(
        depth_small, "DEPTH", (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
    )

    x_off = w - pip_w - 5
    y_off = h - pip_h - 5
    frame[y_off:y_off + pip_h, x_off:x_off + pip_w] = depth_small
    return frame


def _draw_global_hud(
    frame: np.ndarray,
    detections: list[dict],
    lanes_ok: bool,
) -> np.ndarray:
    """Draw title bar and status at the top of the frame."""
    h, w = frame.shape[:2]

    # ── Top bar background ────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────
    cv2.putText(
        frame, "AUTONOMOUS PERCEPTION STACK",
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 200, 255), 2, cv2.LINE_AA
    )

    # ── Status indicators ─────────────────────────────────────────────────
    distances = [d["distance_m"] for d in detections if d["distance_m"]]
    danger    = any(dm < 15.0 for dm in distances)

    lane_col   = (0, 255, 100)  if lanes_ok else (0, 80, 255)
    lane_txt   = "LANE ✓"       if lanes_ok else "LANE ✗"
    danger_col = (0, 0, 255)    if danger   else (0, 200, 100)
    danger_txt = "! DANGER"     if danger   else "SAFE"

    status_x = w - 260
    cv2.putText(frame, lane_txt,   (status_x,       26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lane_col,   2, cv2.LINE_AA)
    cv2.putText(frame, danger_txt, (status_x + 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, danger_col, 2, cv2.LINE_AA)

    return frame


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Perception Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file, or camera index (e.g. 0)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation (much faster)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (default: all)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    args = parser.parse_args()

    pipeline = PerceptionPipeline(
        skip_depth  = args.skip_depth,
        yolo_conf   = args.conf,
        use_wandb   = not args.no_wandb,
        output_dir  = args.output,
    )

    out = pipeline.run(
        source     = args.input,
        show       = args.show,
        max_frames = args.max_frames,
    )

    print(f"\n✅ Done! Output video: {out}")


if __name__ == "__main__":
    main()
