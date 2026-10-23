"""
Tests for Object Detection + Tracking Module
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.detection.yolo_tracker import ObjectTracker


@pytest.fixture(scope="module")
def tracker():
    """Load tracker once for the whole module — avoids re-downloading weights."""
    return ObjectTracker(conf=0.25)


def make_blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_tracker_loads(tracker):
    """ObjectTracker should initialise without error."""
    assert tracker is not None
    assert tracker.model is not None


def test_process_frame_returns_tuple(tracker):
    """process_frame should return (annotated_frame, list_of_dicts)."""
    frame = make_blank_frame()
    result = tracker.process_frame(frame)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_process_frame_annotated_shape(tracker):
    """Annotated frame should be same shape as input."""
    frame = make_blank_frame(480, 640)
    annotated, _ = tracker.process_frame(frame)
    assert annotated.shape == frame.shape


def test_process_frame_detections_list(tracker):
    """Detections should be a list (possibly empty on a blank frame)."""
    frame = make_blank_frame()
    _, detections = tracker.process_frame(frame)
    assert isinstance(detections, list)


def test_detection_dict_keys(tracker):
    """Each detection dict should have required keys."""
    required = {"track_id", "class_name", "bbox", "confidence", "distance_m"}
    frame = make_blank_frame()
    # inject a small fake white rectangle to give YOLO something to look at
    frame[200:280, 250:380] = 255
    _, detections = tracker.process_frame(frame)
    for d in detections:
        assert required.issubset(d.keys()), f"Missing keys in: {d.keys()}"


def test_process_frame_with_depth_map(tracker):
    """Should not crash when a depth map is provided."""
    frame = make_blank_frame()
    depth = np.random.rand(480, 640).astype(np.float32)
    annotated, dets = tracker.process_frame(frame, depth_map=depth)
    assert annotated.shape == frame.shape
    assert isinstance(dets, list)
