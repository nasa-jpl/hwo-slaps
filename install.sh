#!/bin/bash
# HWO-SLAPS Installation Script

echo "================================================"
echo "     HWO-SLAPS Installation Script v1.0         "
echo "  HWO Strong Lensing and PSF Stability Pipeline "
echo "================================================"
echo ""

# Global variable to track installation state
INSTALL_STATE_FILE="/tmp/hwoslaps_install_state"

# Function to log installation progress
log_progress() {
    echo "$1" >> "$INSTALL_STATE_FILE"
}

# Function to check if step was already completed
step_completed() {
    if [ -f "$INSTALL_STATE_FILE" ]; then
        grep -q "$1" "$INSTALL_STATE_FILE"
        return $?
    fi
    return 1
}

# Function to check if command was successful with error recovery
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 successful"
        log_progress "$1"
    else
        echo "✗ $1 failed"
        echo ""
        echo "Installation failed at step: $1"
        echo "You can retry the installation by running this script again."
        echo "The script will skip completed steps and continue from where it failed."
        echo ""
        echo "If the problem persists, you can manually complete the remaining steps:"
        echo "1. Activate the environment: conda activate hwo-slaps"
        echo "2. Install remaining packages manually"
        echo ""
        exit 1
    fi
}

# Function to clean up on successful completion
cleanup_state() {
    if [ -f "$INSTALL_STATE_FILE" ]; then
        rm "$INSTALL_STATE_FILE"
    fi
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "✗ Conda not found. Please install Anaconda or Miniconda first."
    echo "  Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if hwo-slaps environment already exists or was already created
if conda env list | grep -q "^hwo-slaps " || step_completed "Environment creation"; then
    echo "→ Found existing 'hwo-slaps' environment, skipping creation..."
else
    echo "→ Creating conda environment 'hwo-slaps' with Python 3.11..."
    conda create -n hwo-slaps python=3.11 -y
    check_status "Environment creation"
fi

echo ""
echo "→ Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hwo-slaps
check_status "Environment activation"

# Verify we're in the correct environment
if [ "$CONDA_DEFAULT_ENV" != "hwo-slaps" ]; then
    echo "✗ Failed to activate hwo-slaps environment"
    echo "  Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo ""
if ! step_completed "Pip upgrade"; then
    echo "→ Upgrading pip..."
    pip install --upgrade pip
    check_status "Pip upgrade"
else
    echo "→ Pip upgrade already completed, skipping..."
fi

echo ""
if ! step_completed "PyAutoLens installation"; then
    echo "→ Installing PyAutoLens..."
    pip install autolens --no-cache-dir
    check_status "PyAutoLens installation"
else
    echo "→ PyAutoLens already installed, skipping..."
fi

echo ""
if ! step_completed "Numba installation"; then
    echo "→ Installing numba for performance..."
    pip install numba --no-cache-dir
    check_status "Numba installation"
else
    echo "→ Numba already installed, skipping..."
fi

echo ""
if ! step_completed "HCIPy installation"; then
    echo "→ Installing HCIPy..."
    pip install hcipy
    check_status "HCIPy installation"
else
    echo "→ HCIPy already installed, skipping..."
fi

echo ""
if ! step_completed "Additional dependencies"; then
    echo "→ Installing additional dependencies..."
    pip install pyyaml matplotlib numpy scipy astropy tqdm
    check_status "Additional dependencies"
else
    echo "→ Additional dependencies already installed, skipping..."
fi

echo ""
if ! step_completed "Import test"; then
    echo "→ Testing imports..."
    python -c "
import autolens
import hcipy
import numpy
import yaml
print('✓ All imports successful!')
"
    check_status "Import test"
else
    echo "→ Import test already completed, skipping..."
fi

echo ""
if ! step_completed "HWO-SLAPS installation"; then
    echo "→ Installing HWO-SLAPS in development mode..."
    pip install -e .
    check_status "HWO-SLAPS installation"
else
    echo "→ HWO-SLAPS already installed, skipping..."
fi

# Clean up state file on successful completion
cleanup_state

echo ""
echo "================================================"
echo "     ✓ Installation Complete!                   "
echo "================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "    conda activate hwo-slaps"
echo ""
echo "To test the installation, run:"
echo "    python tests/test_installation.py"
echo ""