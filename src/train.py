import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from .config import DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, MODELS_DIR

USE_AMP = DEVICE.type == "cuda"


def train_model(model, train_loader, val_loader, class_weights, model_name="model"):
    """Train model with early stopping, mixed precision (CUDA only), and checkpoint saving."""
    model = model.to(DEVICE)

    weights_tensor = torch.tensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_best.pth")
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return model, history
