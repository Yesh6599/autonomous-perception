"""
Monocular Depth Estimation Module
===================================
Uses MiDaS (Small variant — fast on CPU) to estimate per-pixel depth
from a single RGB image.

Output:
  - depth_map: normalised float32 array (0=close, 1=far)
  - depth_colored: colorised heatmap for visualisation
"""

import cv2
import numpy as np
import torch


class DepthEstimator:
    """
    Wraps the MiDaS small model for monocular depth estimation.
    Optimised for CPU usage with the lightweight DPT_Small backbone.
    """

    def __init__(self, model_type: str = "MiDaS_small"):
        """
        Args:
            model_type: One of "MiDaS_small" (fast), "DPT_Hybrid", "DPT_Large"
                        Use MiDaS_small for CPU — best speed/quality tradeoff.
        """
        print(f"[Depth] Loading MiDaS model: {model_type}")
        self.device = torch.device("cpu")

        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            model_type,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Load matching transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print("[Depth] MiDaS ready ✓")

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth for a single BGR frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            depth_map:     Float32 array (H, W) normalised to [0, 1].
                           Higher = farther away.
            depth_colored: BGR heatmap (H, W, 3) for visualisation.
        """
        h, w = frame.shape[:2]

        # ── Pre-process ───────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        # ── Inference ─────────────────────────────────────────────────────
        prediction = self.model(input_batch)

        # ── Post-process ──────────────────────────────────────────────────
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_raw = prediction.cpu().numpy()

        # MiDaS outputs inverse depth — normalise to [0,1]
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw)

        # Invert so that 0 = close, 1 = far
        depth_map = 1.0 - depth_norm.astype(np.float32)

        # ── Colourmap ─────────────────────────────────────────────────────
        depth_uint8   = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        return depth_map, depth_colored


def blend_depth_overlay(
    frame: np.ndarray,
    depth_colored: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Blend the depth heatmap onto the original frame.

    Args:
        frame:         Original BGR frame
        depth_colored: Colourised depth map (same size as frame)
        alpha:         Opacity of depth overlay (0 = invisible, 1 = full)

    Returns:
        Blended BGR frame
    """
    return cv2.addWeighted(frame, 1.0 - alpha, depth_colored, alpha, 0)
