@echo off
echo ========================================
echo  Autonomous Driving Perception Stack
echo  Quick Setup ^& Run Script
echo ========================================
echo.

:: Step 1: Create virtual environment if not exists
if not exist venv (
    echo [1/4] Creating virtual environment...
    python -m venv venv
)

:: Step 2: Activate
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

:: Step 3: Install dependencies
echo [3/4] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet

:: Step 4: Download sample video
echo [4/4] Downloading sample video...
python data/download_sample.py

echo.
echo ========================================
echo  Setup complete! Running pipeline...
echo ========================================
echo.
echo NOTE: First run downloads YOLOv8 + MiDaS weights (~100MB)
echo.

:: Run pipeline (fast mode - no depth for CPU)
python pipeline/run_pipeline.py --input data/sample_driving.mp4 --skip-depth --show --output output/

echo.
echo Done! Check output/annotated_output.mp4 for the result.
pause
