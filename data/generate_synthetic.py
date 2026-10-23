"""
Synthetic Driving Video Generator
====================================
Creates a realistic synthetic dashcam-style video for testing the pipeline.
Includes:
  - Moving road with lane markings
  - Animated cars (rectangles) of various sizes/speeds
  - A pedestrian crossing simulation
  - Day/night cycle tint
No external downloads needed — pure OpenCV.
"""

import cv2
import numpy as np
import random
import math

W, H   = 1280, 720
FPS    = 30
FRAMES = 600   # 20 seconds

OUT = r"C:\Users\yasha\.gemini\antigravity\scratch\autonomous-perception\data\sample_driving.mp4"

def draw_road(frame, offset):
    """Draw road surface with dashed centre line and solid edges."""
    # Sky gradient
    for y in range(H // 2):
        t = y / (H // 2)
        c = int(135 * (1 - t) + 80 * t)
        frame[y, :] = (c + 20, c + 10, c)

    # Road surface
    frame[H // 2:, :] = (60, 65, 60)

    # Vanishing point
    vp_x, vp_y = W // 2, H // 2

    # Left & right solid white edges
    cv2.line(frame, (vp_x - 30, vp_y + 5), (80, H), (220, 220, 220), 3)
    cv2.line(frame, (vp_x + 30, vp_y + 5), (W - 80, H), (220, 220, 220), 3)

    # Dashed centre line — scrolling
    num_dashes = 12
    for i in range(num_dashes):
        t0 = (i / num_dashes + offset / FRAMES) % 1.0
        t1 = ((i + 0.5) / num_dashes + offset / FRAMES) % 1.0
        if t0 > t1:
            continue
        x0 = int(vp_x + (0) * t0)
        y0 = int(vp_y + (H - vp_y) * t0)
        x1 = int(vp_x + (0) * t1)
        y1 = int(vp_y + (H - vp_y) * t1)
        lw = max(1, int(3 * t0))
        cv2.line(frame, (x0, y0), (x1, y1), (220, 220, 180), lw)

    # Ground texture lines
    for i in range(6):
        t = 0.3 + i * 0.12
        y_line = int(vp_y + (H - vp_y) * t)
        x_left  = int(vp_x + (80  - vp_x) * t)
        x_right = int(vp_x + (W - 80 - vp_x) * t)
        cv2.line(frame, (x_left, y_line), (x_right, y_line), (55, 58, 55), 1)


class Vehicle:
    COLORS = [
        (180, 50,  50),   # red
        (50,  50,  180),  # blue
        (50,  150, 50),   # green
        (200, 200, 50),   # yellow
        (150, 80,  200),  # purple
        (200, 200, 200),  # white
        (40,  40,  40),   # black
    ]

    def __init__(self, lane: int):
        self.lane   = lane          # -1=left, 0=center, 1=right
        self.z      = random.uniform(0.05, 0.8)   # depth (0=far, 1=close)
        self.speed  = random.uniform(0.003, 0.010)
        self.color  = random.choice(self.COLORS)
        self.width  = random.randint(60, 110)
        self.height = random.randint(35, 65)
        self.kind   = random.choice(["car", "truck", "van"])
        if self.kind == "truck":
            self.width  = int(self.width * 1.5)
            self.height = int(self.height * 1.3)

    def update(self):
        self.z += self.speed
        if self.z > 1.05:
            self.z     = random.uniform(0.0, 0.1)
            self.lane  = random.choice([-1, 0, 1])
            self.color = random.choice(self.COLORS)
            self.speed = random.uniform(0.003, 0.010)

    def draw(self, frame):
        vp_x, vp_y = W // 2, H // 2
        scale = self.z ** 1.2

        # Lateral offset per lane
        lane_offset = self.lane * int(200 * scale)
        cx = vp_x + lane_offset
        cy = int(vp_y + (H - vp_y) * self.z)

        bw = int(self.width  * scale)
        bh = int(self.height * scale)

        if bw < 5 or bh < 4:
            return None

        x1, y1 = cx - bw // 2, cy - bh
        x2, y2 = cx + bw // 2, cy

        if x2 <= 0 or x1 >= W or y2 <= vp_y:
            return None

        # Body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # Roof
        roof_h = int(bh * 0.4)
        rx1 = x1 + bw // 6
        rx2 = x2 - bw // 6
        cv2.rectangle(frame, (rx1, y1 - roof_h), (rx2, y1), self.color, -1)
        # Windshield highlight
        cv2.rectangle(frame, (rx1 + 2, y1 - roof_h + 2), (rx2 - 2, y1 - 2), (180, 210, 230), -1)
        # Wheels
        wr = max(2, bw // 8)
        for wx in [x1 + wr, x2 - wr]:
            cv2.circle(frame, (wx, y2), wr, (20, 20, 20), -1)
        # Headlights
        light_size = max(2, bw // 10)
        cv2.rectangle(frame, (x1, y2 - light_size * 2), (x1 + light_size, y2 - light_size), (255, 255, 200), -1)
        cv2.rectangle(frame, (x2 - light_size, y2 - light_size * 2), (x2, y2 - light_size), (200, 50, 50), -1)

        return (x1, y1 - roof_h, x2, y2)   # full bbox


class Pedestrian:
    def __init__(self):
        self.x     = random.choice([200, W - 200])
        self.y     = H - 100
        self.dx    = random.choice([-2, -1, 1, 2])
        self.color = (random.randint(100, 220), random.randint(60, 180), random.randint(80, 200))

    def update(self):
        self.x += self.dx
        if self.x < 50 or self.x > W - 50:
            self.dx *= -1

    def draw(self, frame):
        x, y = int(self.x), int(self.y)
        # Body
        cv2.rectangle(frame, (x - 10, y - 40), (x + 10, y), self.color, -1)
        # Head
        cv2.circle(frame, (x, y - 50), 12, self.color, -1)
        # Legs (animated)
        cv2.line(frame, (x, y), (x - 8, y + 30), (50, 50, 80), 3)
        cv2.line(frame, (x, y), (x + 8, y + 30), (50, 50, 80), 3)


def generate():
    vehicles    = [Vehicle(lane) for lane in ([-1, 0, 1] * 4)]
    pedestrians = [Pedestrian() for _ in range(3)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

    print(f"Generating {FRAMES} frames ({FRAMES // FPS}s) …")
    for f in range(FRAMES):
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        draw_road(frame, f)

        # Sort vehicles back-to-front (painters algorithm)
        vehicles.sort(key=lambda v: v.z)
        for v in vehicles:
            v.update()
            v.draw(frame)

        for p in pedestrians:
            p.update()
            p.draw(frame)

        # Overlay info
        cv2.putText(frame, f"Synthetic Driving Scene — Frame {f+1}/{FRAMES}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if (f + 1) % 60 == 0:
            print(f"  {f+1}/{FRAMES} frames done …")

    writer.release()
    print(f"\n✅ Synthetic video saved → {OUT}")


if __name__ == "__main__":
    generate()
