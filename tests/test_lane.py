"""
Tests for Lane Detection Module
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.lane.opencv_lane import detect_lanes


def make_synthetic_road_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a minimal synthetic road image with white lane lines
    on a grey road surface.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (80, 80, 80)   # grey road

    # Left lane line
    cv2.line(frame, (int(width * 0.25), height), (int(width * 0.40), int(height * 0.55)), (255, 255, 255), 6)
    # Right lane line
    cv2.line(frame, (int(width * 0.75), height), (int(width * 0.60), int(height * 0.55)), (255, 255, 255), 6)
    return frame


def test_detect_lanes_returns_tuple():
    """detect_lanes should return a 2-tuple."""
    frame = make_synthetic_road_frame()
    result = detect_lanes(frame)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_detect_lanes_annotated_frame_shape():
    """Annotated frame should have the same shape as input."""
    frame = make_synthetic_road_frame()
    annotated, _ = detect_lanes(frame)
    assert annotated.shape == frame.shape


def test_detect_lanes_annotated_frame_is_bgr():
    """Annotated frame should be a uint8 BGR array."""
    frame = make_synthetic_road_frame()
    annotated, _ = detect_lanes(frame)
    assert annotated.dtype == np.uint8
    assert annotated.ndim == 3
    assert annotated.shape[2] == 3


def test_detect_lanes_finds_lines_on_synthetic():
    """Should detect at least one lane line on our synthetic frame."""
    frame = make_synthetic_road_frame()
    _, lane_lines = detect_lanes(frame)
    assert len(lane_lines) >= 1, "Expected at least one lane line on synthetic frame"


def test_detect_lanes_black_frame():
    """Should not crash on a fully black frame (no lanes)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, lane_lines = detect_lanes(frame)
    assert annotated.shape == (480, 640, 3)
    assert isinstance(lane_lines, list)


def test_detect_lanes_small_frame():
    """Should handle small frames without error."""
    frame = make_synthetic_road_frame(width=320, height=240)
    annotated, _ = detect_lanes(frame)
    assert annotated.shape == (240, 320, 3)
