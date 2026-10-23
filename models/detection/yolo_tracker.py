"""
Object Detection + Tracking Module
====================================
Uses:
  - YOLOv8n (nano) for fast CPU-friendly detection
  - YOLO built-in ByteTrack for multi-object tracking

Classes tracked: car, truck, bus, motorcycle, person, bicycle
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# ── Class configuration ────────────────────────────────────────────────────
DRIVING_CLASSES = {
    0:  ("person",     (255, 80,  80)),   # red
    1:  ("bicycle",    (255, 160, 0)),    # orange
    2:  ("car",        (80,  180, 255)),  # blue
    3:  ("motorcycle", (200, 80,  255)),  # purple
    5:  ("bus",        (80,  255, 180)),  # cyan
    7:  ("truck",      (255, 220, 0)),    # yellow
}


class ObjectTracker:
    """
    Wraps YOLOv8n + YOLO's built-in ByteTrack into a single object that:
      - detects + tracks objects in a frame
      - assigns persistent track IDs across frames
      - returns annotated frame + structured detection list
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        print(f"[Tracker] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf

        # Annotators (supervision 0.31.1 compatible)
        self.box_annotator   = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.55,
            text_thickness=1,
            text_padding=4,
        )
        print("[Tracker] YOLOv8n + ByteTrack ready (built-in tracker)")

    def process_frame(
        self, frame: np.ndarray, depth_map: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run detection + tracking on a single frame.
        Uses YOLO's native .track() which uses ByteTrack internally.
        """
        h, w = frame.shape[:2]

        # ── 1. YOLO track (ByteTrack built-in) ────────────────────────────
        results = self.model.track(
            frame,
            conf=self.conf,
            classes=list(DRIVING_CLASSES.keys()),
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]

        # ── 2. Parse results manually ──────────────────────────────────────
        detections_info = []
        labels          = []
        annotated       = frame.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes      = results.boxes.xyxy.cpu().numpy()
            class_ids  = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            track_ids  = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else [-1] * len(boxes)
            )

            for i in range(len(boxes)):
                bbox       = boxes[i].tolist()
                class_id   = int(class_ids[i])
                confidence = float(confidences[i])
                track_id   = int(track_ids[i])

                class_name, color = DRIVING_CLASSES.get(
                    class_id, ("object", (200, 200, 200))
                )

                # ── Depth sampling ────────────────────────────────────────
                distance_m = None
                if depth_map is not None:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))
                    raw_depth  = float(depth_map[cy, cx])
                    distance_m = round(2.0 + (1.0 - raw_depth) * 78.0, 1)

                detections_info.append({
                    "track_id":   track_id,
                    "class_name": class_name,
                    "class_id":   class_id,
                    "bbox":       bbox,
                    "confidence": round(confidence, 3),
                    "distance_m": distance_m,
                })

                # ── Draw box ──────────────────────────────────────────────
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                dist_str = f" {distance_m}m" if distance_m else ""
                label    = f"#{track_id} {class_name}{dist_str} {confidence:.0%}"
                label_y  = max(y1 - 8, 15)

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
                cv2.putText(annotated, label, (x1 + 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

                # ── Danger alert ──────────────────────────────────────────
                if distance_m is not None and distance_m < 15.0:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"! CLOSE {distance_m}m",
                                (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # ── 3. HUD overlay ─────────────────────────────────────────────────
        _draw_hud(annotated, detections_info)

        return annotated, detections_info


def _draw_hud(frame: np.ndarray, detections_info: list[dict]) -> None:
    """Draw object count HUD in top-right corner."""
    h, w = frame.shape[:2]
    counts = {}
    for d in detections_info:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

    hud_lines = [f"Objects: {len(detections_info)}"] + [
        f"  {k}: {v}" for k, v in sorted(counts.items())
    ]

    x_start = w - 200
    y_start = 48  # below global HUD bar

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x_start - 10, y_start - 15),
                  (w - 5, y_start + len(hud_lines) * 22),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line,
                    (x_start, y_start + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

