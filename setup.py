#!/usr/bin/env python3
"""
Setup script for YOLOv8 Object Detection System
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("This system requires Python 3.8 or higher")
        return False
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = ["data", "outputs", "models"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True


def download_model():
    """Download YOLOv8 model."""
    print("\nDownloading YOLOv8 model...")
    
    try:
        # Import and download model
        script = """
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from ultralytics import YOLO
print("Downloading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("Model downloaded successfully!")
"""
        
        subprocess.check_call([
            sys.executable, "-c", script
        ])
        print("✓ YOLOv8n model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download model: {e}")
        print("The model will be downloaded automatically on first use")
        return False


def run_tests():
    """Run system tests."""
    print("\nRunning system tests...")
    
    try:
        subprocess.check_call([
            sys.executable, "test_system.py"
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False


def main():
    """Main setup function."""
    print("YOLOv8 Object Detection System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nTroubleshooting:")
        print("1. Try: pip install --upgrade pip")
        print("2. Try: pip install -r requirements.txt --user")
        print("3. Check your internet connection")
        sys.exit(1)
    
    # Download model (optional)
    download_model()
    
    # Run tests
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Place images/videos in the 'data/' directory")
    print("2. Run: python app.py --input data/image.jpg --output outputs/result.jpg")
    print("3. Or run: python example.py for examples")
    print("4. Or run: python test_system.py to verify everything works")
    
    print("\nQuick start examples:")
    print("  # Test the system")
    print("  python test_system.py")
    print("")
    print("  # Run examples")
    print("  python example.py")
    print("")
    print("  # Use webcam")
    print("  python app.py --webcam")
    print("")
    print("  # Process an image")
    print("  python app.py --input data/your_image.jpg --output outputs/result.jpg")


if __name__ == "__main__":
    main() 