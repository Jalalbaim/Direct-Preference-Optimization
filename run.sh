#!/bin/bash

# DPO vs PPO Sentiment Generation Experiments
# Quick setup and run script

set -e

echo "======================================================================"
echo "DPO vs PPO: Sentiment Generation Experiments"
echo "Paper: Direct Preference Optimization"
echo "======================================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

echo "✓ Python: $(python3 --version)"

# Check PyTorch device info
echo ""
echo "Checking device availability..."
python3 device_info.py

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Run full pipeline
echo ""
echo "Starting DPO vs PPO experiments..."
echo ""

python3 main.py --stage all

echo ""
echo "======================================================================"
echo "Experiments completed! Check ./results/ for trained models and plots."
echo "======================================================================"
