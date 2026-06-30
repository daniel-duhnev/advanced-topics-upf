import torch.nn as nn
from torchvision import models


def get_resnet50(num_classes=7, pretrained=True):
    """ResNet-50 with frozen early layers and custom classifier head."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze all layers except layer4 and fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def get_efficientnet_b0(num_classes=7, pretrained=True):
    """EfficientNet-B0 with frozen early layers and custom classifier head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze all features except last block
    for name, param in model.features[:-2].named_parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model
