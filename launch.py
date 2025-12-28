#!/usr/bin/env python3
"""
Launch script for MiraTTS Web Interface
Simple wrapper to start the web UI with common configurations
"""

import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch MiraTTS Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--model", default="YatharthS/MiraTTS", help="Model path or HF model ID")
    
    args = parser.parse_args()
    
    cmd = [
        sys.executable, "web_ui.py",
        "--server_name", args.host,
        "--server_port", str(args.port),
        "--model_dir", args.model
    ]
    
    if args.share:
        cmd.append("--share")
    
    print(f"Launching MiraTTS Web Interface...")
    print(f"Model: {args.model}")
    print(f"URL: http://{args.host}:{args.port}")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()