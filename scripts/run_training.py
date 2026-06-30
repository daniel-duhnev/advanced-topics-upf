"""
Standalone training script — runs the full pipeline from terminal.

Usage:
    conda activate skin-lesion
    python scripts/run_training.py              # Full training (25 epochs, early stopping)
    python scripts/run_training.py --epochs 2   # Quick test (2 epochs)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import random
import json


def main():
    parser = argparse.ArgumentParser(description="Train skin lesion classifiers")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to HAM10000 data")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0", "both"], default="both")
    args = parser.parse_args()

    # Override epochs if specified
    from src import config
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    from src.config import DEVICE, MODELS_DIR, RESULTS_DIR, SEED, DATA_DIR
    from src.dataset import get_dataloaders
    from src.models import get_resnet50, get_efficientnet_b0
    from src.train import train_model
    from src.evaluate import evaluate_model, compute_metrics, plot_confusion_matrix, plot_training_curves

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 60)
    print("SKIN LESION CLASSIFICATION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device:     {DEVICE}")
    print(f"Epochs:     {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print()

    # Resolve data directory
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        # Try looking one level up
        for candidate in [data_dir, data_dir.parent]:
            if (candidate / "HAM10000_metadata.csv").exists():
                metadata_path = candidate / "HAM10000_metadata.csv"
                data_dir = candidate
                break
        else:
            print(f"ERROR: Cannot find HAM10000_metadata.csv in {data_dir}")
            print("Run ./scripts/setup_local.sh first to download the dataset.")
            sys.exit(1)

    print(f"Data dir:   {data_dir}")
    print(f"Metadata:   {metadata_path}")
    print()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(metadata_path, data_dir)
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val:   {len(val_loader.dataset)} images")
    print(f"  Test:  {len(test_loader.dataset)} images")
    print(f"  Class weights: {class_weights}")
    print()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results = {}

    # Train ResNet-50
    if args.model in ("resnet50", "both"):
        print("=" * 60)
        print("TRAINING: ResNet-50")
        print("=" * 60)
        resnet = get_resnet50(num_classes=7, pretrained=True)
        trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")
        print()

        resnet, resnet_history = train_model(resnet, train_loader, val_loader, class_weights, model_name="resnet50")

        print("\nEvaluating ResNet-50 on test set...")
        resnet.load_state_dict(torch.load(MODELS_DIR / "resnet50_best.pth", map_location=DEVICE))
        preds, labels = evaluate_model(resnet, test_loader)
        metrics = compute_metrics(labels, preds)
        results["resnet50"] = metrics

        plot_confusion_matrix(labels, preds, title="ResNet-50", save_path=RESULTS_DIR / "confusion_matrix_resnet50.png")
        plot_training_curves(resnet_history, title="ResNet-50", save_path=RESULTS_DIR / "training_curves_resnet50.png")

    # Train EfficientNet-B0
    if args.model in ("efficientnet_b0", "both"):
        print("\n" + "=" * 60)
        print("TRAINING: EfficientNet-B0")
        print("=" * 60)
        effnet = get_efficientnet_b0(num_classes=7, pretrained=True)
        trainable = sum(p.numel() for p in effnet.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")
        print()

        effnet, effnet_history = train_model(effnet, train_loader, val_loader, class_weights, model_name="efficientnet_b0")

        print("\nEvaluating EfficientNet-B0 on test set...")
        effnet.load_state_dict(torch.load(MODELS_DIR / "efficientnet_b0_best.pth", map_location=DEVICE))
        preds, labels = evaluate_model(effnet, test_loader)
        metrics = compute_metrics(labels, preds)
        results["efficientnet_b0"] = metrics

        plot_confusion_matrix(labels, preds, title="EfficientNet-B0", save_path=RESULTS_DIR / "confusion_matrix_efficientnet.png")
        plot_training_curves(effnet_history, title="EfficientNet-B0", save_path=RESULTS_DIR / "training_curves_efficientnet.png")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for model_name, m in results.items():
        print(f"  {model_name:20s} | Acc: {m['accuracy']:.4f} | F1 macro: {m['f1_macro']:.4f} | F1 weighted: {m['f1_weighted']:.4f}")

    # Save results JSON
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Models saved to {MODELS_DIR}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
