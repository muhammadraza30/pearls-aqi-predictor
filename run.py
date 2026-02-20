"""
Unified launcher — starts both FastAPI and Streamlit with a single command.

Usage:
    python run.py            # Launch both services
    python run.py --api      # FastAPI only
    python run.py --ui       # Streamlit only
"""
import subprocess
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def start_fastapi():
    """Start FastAPI server on port 8000."""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )


def start_streamlit():
    """Start Streamlit dashboard on port 8501."""
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch AQI Predictor services")
    parser.add_argument("--api", action="store_true", help="Start FastAPI only")
    parser.add_argument("--ui", action="store_true", help="Start Streamlit only")
    args = parser.parse_args()

    # If neither flag, start both
    start_api = args.api or (not args.api and not args.ui)
    start_ui = args.ui or (not args.api and not args.ui)

    procs = []
    try:
        if start_api:
            print("🚀 Starting FastAPI on http://localhost:8000 ...")
            procs.append(start_fastapi())
            time.sleep(1)

        if start_ui:
            print("🎨 Starting Streamlit on http://localhost:8501 ...")
            procs.append(start_streamlit())

        print("\n✅ All services running! Press Ctrl+C to stop.\n")
        # Wait for all processes
        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        print("Done.")


if __name__ == "__main__":
    main()
