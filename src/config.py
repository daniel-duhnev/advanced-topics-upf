from pathlib import Path
import torch

# --- Paths ---
# Adjust DATA_DIR based on environment (Colab vs local)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "HAM10000"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Dataset ---
NUM_CLASSES = 7
IMAGE_SIZE = 224
CLASS_NAMES = {
    0: "akiec",  # Actinic Keratoses
    1: "bcc",    # Basal Cell Carcinoma
    2: "bkl",    # Benign Keratosis
    3: "df",     # Dermatofibroma
    4: "mel",    # Melanoma
    5: "nv",     # Melanocytic Nevi
    6: "vasc",   # Vascular Lesions
}
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Bowen's Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 1e-4
SEED = 42

# --- Data Split ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Normalization (ImageNet) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Device (CUDA > MPS > CPU) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
