#!/bin/bash
# Setup script for local execution on macOS (Apple Silicon)
# Run once: ./scripts/setup_local.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/HAM10000"

echo "=== Skin Lesion Classification — Local Setup ==="
echo "Project: $PROJECT_DIR"
echo ""

# --- Step 1: Create conda environment ---
echo "[1/3] Creating conda environment 'skin-lesion'..."
if conda env list | grep -q "skin-lesion"; then
    echo "  Environment already exists. Skipping."
else
    conda env create -f "$PROJECT_DIR/environment.yml"
    echo "  Done."
fi

# --- Step 2: Download dataset ---
echo ""
echo "[2/3] Downloading HAM10000 dataset..."

if [ -d "$DATA_DIR" ] && [ "$(find "$DATA_DIR" -name '*.jpg' | head -1)" ]; then
    echo "  Dataset already exists at $DATA_DIR. Skipping."
else
    # Check for kaggle CLI
    if ! command -v kaggle &> /dev/null; then
        echo "  Installing kaggle CLI..."
        conda run -n skin-lesion pip install kaggle
    fi

    # Check for credentials (supports both kaggle.json and access_token)
    if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ~/.kaggle/access_token ]; then
        echo ""
        echo "  ERROR: Kaggle credentials not found."
        echo ""
        echo "  To fix (option A — API token):"
        echo "  1. Go to https://www.kaggle.com/settings → API → Create New Token"
        echo "  2. Run: mkdir -p ~/.kaggle && echo 'YOUR_TOKEN' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token"
        echo ""
        echo "  To fix (option B — kaggle.json):"
        echo "  1. Download kaggle.json from Kaggle settings"
        echo "  2. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "  Then re-run this script."
        exit 1
    fi

    mkdir -p "$DATA_DIR"
    echo "  Downloading from Kaggle (~3 GB)..."
    conda run -n skin-lesion kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$DATA_DIR"
    echo "  Extracting..."
    unzip -q "$DATA_DIR/skin-cancer-mnist-ham10000.zip" -d "$DATA_DIR"
    rm -f "$DATA_DIR/skin-cancer-mnist-ham10000.zip"
    echo "  Done."
fi

# --- Step 3: Verify ---
echo ""
echo "[3/3] Verifying setup..."
IMG_COUNT=$(find "$DATA_DIR" -name "*.jpg" | wc -l | tr -d ' ')
echo "  Images found: $IMG_COUNT"
if [ "$IMG_COUNT" -lt 10000 ]; then
    echo "  WARNING: Expected ~10,015 images but found $IMG_COUNT"
else
    echo "  Dataset looks good."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To train:"
echo "  conda activate skin-lesion"
echo "  cd $PROJECT_DIR"
echo "  python scripts/run_training.py"
echo ""
echo "Or use Jupyter:"
echo "  conda activate skin-lesion"
echo "  jupyter notebook notebooks/02_train_and_evaluate.ipynb"
