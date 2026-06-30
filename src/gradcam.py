import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from .config import DEVICE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES, IMAGE_SIZE


class GradCAM:
    """Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)."""

    def __init__(self, model, target_layer):
        self.model = model.eval().to(DEVICE)
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for an input image tensor."""
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE) if input_tensor.dim() == 3 else input_tensor.to(DEVICE)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


def get_target_layer(model, model_name):
    """Get the appropriate target layer for Grad-CAM based on architecture."""
    if "resnet" in model_name.lower():
        return model.layer4[-1]
    elif "efficientnet" in model_name.lower():
        return model.features[-1]
    raise ValueError(f"Unknown model: {model_name}")


def visualize_gradcam(image_path, model, model_name, true_label=None):
    """Generate and display Grad-CAM visualization for a single image."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img)

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_class = gradcam.generate(input_tensor)

    # Resize heatmap to image size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )) / 255.0

    # Create overlay
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array)
    axes[0].set_title(f"Original (True: {CLASS_NAMES.get(true_label, '?')})")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM (Pred: {CLASS_NAMES[pred_class]})")
    axes[1].axis("off")

    overlay = img_array * 0.5 + plt.cm.jet(heatmap_resized)[:, :, :3] * 0.5
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, pred_class
