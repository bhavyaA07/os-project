#!/bin/bash
echo "==================================================="
echo "  Setting up and Starting CPU Scheduling Dashboard"
echo "==================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Creating new virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[INFO] Installing required packages from requirements.txt..."
pip install -r requirements.txt

# Run the application
echo "[INFO] Starting Dashboard Server..."
python3 app.py
