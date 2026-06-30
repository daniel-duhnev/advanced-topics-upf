"""
Generate Grad-CAM visualisations for both models.

Usage:
    conda activate skin-lesion
    python scripts/run_gradcam.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.config import DEVICE, MODELS_DIR, RESULTS_DIR, DATA_DIR, CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.models import get_resnet50, get_efficientnet_b0
from src.gradcam import GradCAM, get_target_layer


def load_model(model_name):
    if model_name == "resnet50":
        model = get_resnet50(num_classes=7, pretrained=False)
    else:
        model = get_efficientnet_b0(num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(
        MODELS_DIR / f"{model_name}_best.pth", map_location=DEVICE, weights_only=True
    ))
    return model.eval().to(DEVICE)


def generate_gradcam_grid(model, model_name, samples, image_dirs, save_path):
    """Generate a 7x3 grid (original | heatmap | overlay) for one model."""
    label_map = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    fig, axes = plt.subplots(7, 3, figsize=(12, 28))
    fig.suptitle(f"Grad-CAM Attention Maps - {model_name}", fontsize=14, y=1.005)

    target_layer = get_target_layer(model, model_name)

    for row_idx, (_, sample) in enumerate(samples.iterrows()):
        img_path = find_image(sample["image_id"], image_dirs)
        true_label = label_map[sample["dx"]]

        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img)

        gradcam = GradCAM(model, target_layer)
        heatmap, pred_class = gradcam.generate(input_tensor)

        heatmap_resized = np.array(Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)) / 255.0

        img_resized = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

        # Original
        axes[row_idx, 0].imshow(img_resized)
        axes[row_idx, 0].set_title(f"True: {sample['dx']}", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Heatmap
        axes[row_idx, 1].imshow(heatmap_resized, cmap="jet")
        axes[row_idx, 1].set_title(f"Pred: {CLASS_NAMES[pred_class]}", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Overlay
        overlay = img_resized * 0.5 + plt.cm.jet(heatmap_resized)[:, :, :3] * 0.5
        correct = "Y" if pred_class == true_label else "N"
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f"Overlay (correct: {correct})", fontsize=10)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def find_image(image_id, image_dirs):
    for d in image_dirs:
        path = d / f"{image_id}.jpg"
        if path.exists():
            return path
    raise FileNotFoundError(f"{image_id}.jpg not found")


def main():
    print("=" * 60)
    print("GRAD-CAM VISUALISATIONS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # Find images
    image_dirs = []
    for subdir in ["images", "HAM10000_images_part_1", "HAM10000_images_part_2"]:
        p = DATA_DIR / subdir
        if p.is_dir():
            image_dirs.append(p)
    if not image_dirs:
        if any(DATA_DIR.glob("*.jpg")):
            image_dirs = [DATA_DIR]
    print(f"Image dirs: {image_dirs}")

    # Select one representative image per class
    metadata = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")
    samples = metadata.groupby("dx").sample(1, random_state=42).reset_index(drop=True)
    print(f"Selected {len(samples)} images (one per class):")
    print(samples[["image_id", "dx"]].to_string(index=False))
    print()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Run Grad-CAM on both models
    for model_name in ["resnet50", "efficientnet_b0"]:
        print(f"Generating Grad-CAM for {model_name}...")
        model = load_model(model_name)
        save_path = RESULTS_DIR / f"gradcam_{model_name}.png"
        generate_gradcam_grid(model, model_name, samples, image_dirs, save_path)

    print()
    print("Done! Check results/ for the Grad-CAM figures.")


if __name__ == "__main__":
    main()
