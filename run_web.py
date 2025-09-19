#!/usr/bin/env python3
"""
Script untuk menjalankan web dashboard deteksi objek
Usage: python run_web.py
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask_socketio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Dependencies missing:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install dependencies with:")
        print("   pip install -r Requirements.txt")
        return False
    
    return True

def main():
    print("🚀 Starting Object Detection Web Dashboard...")
    print("=" * 50)
    
    # Check if YOLO model exists
    if not os.path.exists("yolov8n.pt"):
        print("⚠️  YOLO model (yolov8n.pt) not found!")
        print("   The model will be downloaded automatically on first run.")
    
    # Check if sample video exists
    if not os.path.exists("sample.mp4"):
        print("⚠️  Sample video (sample.mp4) not found!")
        print("   Make sure to have a video file or use webcam option.")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    print("🌐 Starting web server...")
    print("📱 Dashboard will be available at: http://localhost:9000")
    print("🔴 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the app
        from app import socketio, app
        socketio.run(app, debug=True, host='0.0.0.0', port=9000)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()