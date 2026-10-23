"""
Lane Detection Module
=====================
Detects lane lines using classical OpenCV techniques:
  1. Grayscale + Gaussian Blur
  2. Canny Edge Detection
  3. Region of Interest Masking
  4. Hough Line Transform
  5. Weighted overlay on original frame
"""

import cv2
import numpy as np


def detect_lanes(frame: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Detect lane lines in a single frame.

    Args:
        frame: BGR image as numpy array (H, W, 3)

    Returns:
        annotated_frame: Frame with lane overlay drawn
        lane_lines: List of detected (x1,y1,x2,y2) line segments
    """
    height, width = frame.shape[:2]

    # ── 1. Pre-processing ──────────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 2. Edge Detection ──────────────────────────────────────────────────
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # ── 3. Region of Interest ──────────────────────────────────────────────
    # Trapezoid focused on the lower half of the frame (road area)
    roi_vertices = np.array([[
        (int(width * 0.05), height),           # bottom-left
        (int(width * 0.42), int(height * 0.60)),  # top-left
        (int(width * 0.58), int(height * 0.60)),  # top-right
        (int(width * 0.95), height),           # bottom-right
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # ── 4. Hough Line Transform ────────────────────────────────────────────
    raw_lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=80,
        maxLineGap=100,
    )

    # ── 5. Separate & Average Left/Right Lanes ─────────────────────────────
    left_lines, right_lines = [], []
    lane_lines = []

    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line.flatten()[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 == x1:
                continue  # skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.3:          # negative slope → left lane
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:         # positive slope → right lane
                right_lines.append((x1, y1, x2, y2))


    def average_lane(lines, y_start, y_end):
        """Fit a single averaged line through a list of segments."""
        if not lines:
            return None
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        poly = np.polyfit(y_coords, x_coords, 1)   # x = f(y)
        x_start = int(np.polyval(poly, y_start))
        x_end   = int(np.polyval(poly, y_end))
        return (x_start, y_start, x_end, y_end)

    y_bottom = height
    y_top    = int(height * 0.62)

    left_lane  = average_lane(left_lines,  y_bottom, y_top)
    right_lane = average_lane(right_lines, y_bottom, y_top)

    for lane in [left_lane, right_lane]:
        if lane is not None:
            lane_lines.append(lane)

    # ── 6. Draw Overlay ────────────────────────────────────────────────────
    annotated = _draw_lanes(frame, left_lane, right_lane, height)
    return annotated, lane_lines


def _draw_lanes(
    frame: np.ndarray,
    left_lane,
    right_lane,
    height: int,
) -> np.ndarray:
    """Draw filled lane polygon and lane lines onto frame."""
    overlay = frame.copy()
    line_img = np.zeros_like(frame)

    # Draw individual lane lines
    lane_color = (0, 255, 100)   # bright green
    thickness  = 6

    if left_lane:
        x1, y1, x2, y2 = left_lane
        cv2.line(line_img, (x1, y1), (x2, y2), lane_color, thickness)

    if right_lane:
        x1, y1, x2, y2 = right_lane
        cv2.line(line_img, (x1, y1), (x2, y2), lane_color, thickness)

    # Fill the area between lanes with a semi-transparent tint
    if left_lane and right_lane:
        lx1, ly1, lx2, ly2 = left_lane
        rx1, ry1, rx2, ry2 = right_lane
        poly_pts = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]])
        cv2.fillPoly(line_img, [poly_pts], (0, 120, 255))   # orange tint

    # Blend line_img onto the original frame
    result = cv2.addWeighted(overlay, 1.0, line_img, 0.35, 0)

    # Re-draw sharp lane lines on top of the blend
    if left_lane:
        x1, y1, x2, y2 = left_lane
        cv2.line(result, (x1, y1), (x2, y2), lane_color, thickness)
    if right_lane:
        x1, y1, x2, y2 = right_lane
        cv2.line(result, (x1, y1), (x2, y2), lane_color, thickness)

    # ── Label ──────────────────────────────────────────────────────────────
    label_y = height - 20
    if left_lane and right_lane:
        status = "LANES DETECTED"
        color  = (0, 255, 100)
    elif left_lane or right_lane:
        status = "PARTIAL LANE"
        color  = (0, 200, 255)
    else:
        status = "NO LANE"
        color  = (0, 0, 255)

    cv2.putText(
        result, status, (20, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
    )

    return result
