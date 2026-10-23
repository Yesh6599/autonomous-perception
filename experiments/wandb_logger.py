"""
W&B Experiment Logger
======================
Centralised Weights & Biases logging for the perception pipeline.
Falls back to local CSV logging if W&B is not configured.
"""

import os
import csv
import time
from pathlib import Path
from typing import Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PerceptionLogger:
    """
    Logs per-frame metrics to Weights & Biases (if configured)
    and always writes a local CSV as fallback.
    """

    def __init__(
        self,
        project: str = "autonomous-perception",
        run_name: str | None = None,
        output_dir: str = "output",
        use_wandb: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count   = 0
        self.start_time    = time.time()
        self._fps_buffer   = []
        self._use_wandb    = False

        # ── W&B init ──────────────────────────────────────────────────────
        if use_wandb and WANDB_AVAILABLE:
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                try:
                    wandb.init(
                        project=project,
                        name=run_name or f"run_{int(time.time())}",
                        config={
                            "model":       "yolov8n",
                            "tracker":     "bytetrack",
                            "depth_model": "midas_small",
                            "lane_method": "opencv",
                        },
                    )
                    self._use_wandb = True
                    print("[Logger] W&B logging enabled ✓")
                except Exception as e:
                    print(f"[Logger] W&B init failed ({e}) — using local CSV only")
            else:
                print("[Logger] WANDB_API_KEY not set — using local CSV only")
                print("         Set it with: $env:WANDB_API_KEY='your_key'")
        else:
            print("[Logger] W&B not available — using local CSV only")

        # ── CSV fallback ──────────────────────────────────────────────────
        csv_path = self.output_dir / "metrics.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
            "frame", "timestamp_s", "fps",
            "n_detections", "n_cars", "n_persons", "n_trucks",
            "lanes_detected", "closest_object_m",
        ])
        self._csv_writer.writeheader()
        print(f"[Logger] CSV metrics → {csv_path}")

    def log_frame(
        self,
        detections: list[dict],
        lanes_detected: bool,
        frame_time_s: float,
    ) -> None:
        """
        Log metrics for a single processed frame.

        Args:
            detections:     List of detection dicts from ObjectTracker
            lanes_detected: Whether lanes were found this frame
            frame_time_s:   Wall-clock time to process this frame (seconds)
        """
        self.frame_count += 1
        fps = 1.0 / max(frame_time_s, 1e-6)
        self._fps_buffer.append(fps)
        elapsed = time.time() - self.start_time

        # ── Count by class ────────────────────────────────────────────────
        n_cars    = sum(1 for d in detections if d["class_name"] == "car")
        n_persons = sum(1 for d in detections if d["class_name"] == "person")
        n_trucks  = sum(1 for d in detections if d["class_name"] == "truck")

        distances = [d["distance_m"] for d in detections if d["distance_m"]]
        closest   = round(min(distances), 1) if distances else None

        row = {
            "frame":            self.frame_count,
            "timestamp_s":      round(elapsed, 3),
            "fps":              round(fps, 2),
            "n_detections":     len(detections),
            "n_cars":           n_cars,
            "n_persons":        n_persons,
            "n_trucks":         n_trucks,
            "lanes_detected":   int(lanes_detected),
            "closest_object_m": closest,
        }

        # Write to CSV
        self._csv_writer.writerow(row)

        # Write to W&B
        if self._use_wandb:
            wandb.log(row, step=self.frame_count)

    def log_video(self, video_path: str) -> None:
        """Upload the output video to W&B."""
        if self._use_wandb:
            wandb.log({"demo_video": wandb.Video(video_path, fps=15, format="mp4")})
            print(f"[Logger] Video uploaded to W&B: {video_path}")

    def summary(self) -> dict[str, Any]:
        """Print and return a final summary of the run."""
        avg_fps = sum(self._fps_buffer) / max(len(self._fps_buffer), 1)
        total_s = time.time() - self.start_time
        summary = {
            "total_frames":    self.frame_count,
            "total_time_s":    round(total_s, 2),
            "avg_fps":         round(avg_fps, 2),
            "min_fps":         round(min(self._fps_buffer, default=0), 2),
            "max_fps":         round(max(self._fps_buffer, default=0), 2),
        }
        print("\n" + "=" * 45)
        print("  PIPELINE SUMMARY")
        print("=" * 45)
        for k, v in summary.items():
            print(f"  {k:<22} {v}")
        print("=" * 45)

        if self._use_wandb:
            wandb.summary.update(summary)

        return summary

    def close(self) -> None:
        """Finalise logging and close file handles."""
        self.summary()
        self._csv_file.close()
        if self._use_wandb:
            wandb.finish()
            print("[Logger] W&B run finished.")
