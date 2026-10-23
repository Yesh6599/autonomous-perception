"""
Script to download a free driving sample video for testing.
Uses a Creative Commons dashcam clip from the web.
"""

import urllib.request
import sys
from pathlib import Path

SAMPLE_URL = (
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
)

# Better driving video from a free source
DRIVING_URL = (
    "https://www.pexels.com/download/video/2675139/"
)


def download_sample(out_dir: str = "data") -> str:
    """
    Download a sample driving video for testing the pipeline.
    Falls back to a generic sample if the driving video is unavailable.
    """
    out_path = Path(out_dir) / "sample_driving.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[Download] Already exists: {out_path}")
        return str(out_path)

    # Use yt-dlp if available for a real driving video
    try:
        import subprocess
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]",
                "--merge-output-format", "mp4",
                "-o", str(out_path),
                "https://www.youtube.com/watch?v=nCNS4aaByNA",  # BDD100K demo
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and out_path.exists():
            print(f"[Download] Downloaded via yt-dlp → {out_path}")
            return str(out_path)
    except Exception:
        pass

    # Fallback: generic MP4 sample (not a driving video, just for testing)
    print(f"[Download] Downloading sample video from Google CDN …")
    print(f"[Download] NOTE: This is a generic test video. For real results,")
    print(f"[Download]       place a dashcam MP4 in: {out_dir}/sample_driving.mp4")

    try:
        urllib.request.urlretrieve(
            SAMPLE_URL,
            str(out_path),
            reporthook=lambda b, bs, t: print(
                f"\r  {b * bs / 1e6:.1f} MB", end="", flush=True
            ),
        )
        print(f"\n[Download] Saved → {out_path}")
    except Exception as e:
        print(f"[Download] Failed: {e}")
        print("\n  Please manually place a driving video at:")
        print(f"  {out_path.absolute()}")
        sys.exit(1)

    return str(out_path)


if __name__ == "__main__":
    path = download_sample("data")
    print(f"\nReady to use: {path}")
    print("\nRun the pipeline with:")
    print(f"  python pipeline/run_pipeline.py --input {path} --skip-depth --show")
