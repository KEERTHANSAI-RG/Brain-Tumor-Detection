"""Generate a Grad-CAM explanation for a trained brain MRI classifier."""

import argparse
import pickle
from pathlib import Path

import matplotlib

# This script saves plots and does not need an interactive display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


# Resolve project paths relative to this file, not the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "brain_tumor_resnet18.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

# Keep this order identical to the training script and saved checkpoint.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224
NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

# Evaluation preprocessing matches training's deterministic evaluation transform.
# Random flips and rotations are training-only augmentations, so they are not
# applied to an image when explaining a prediction.
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def validate_mri_image(image: Image.Image) -> tuple[bool, str]:
    """Apply conservative checks for obviously unsuitable MRI inputs.

    This is a lightweight image heuristic, not a medical-image detector. It
    catches strongly colored photographs while allowing grayscale scans and
    colored borders or headers. It cannot reliably identify grayscale photos.
    """
    width, height = image.size
    if min(width, height) < 64:
        return False, "The image resolution is too small to be a useful MRI scan."

    aspect_ratio = width / height
    if not 0.5 <= aspect_ratio <= 2.0:
        return False, "The image proportions are outside the expected MRI scan range."

    # Ignore a narrow border so colored labels around a grayscale scan do not
    # make an otherwise valid scan look like a photograph.
    resized_rgb = np.asarray(
        image.convert("RGB").resize((128, 128), resample=Image.Resampling.BILINEAR),
        dtype=np.int16,
    )
    central_region = resized_rgb[8:-8, 8:-8]
    chroma = central_region.max(axis=2) - central_region.min(axis=2)
    strongly_colored_fraction = float(np.mean(chroma > 32))
    if strongly_colored_fraction > 0.10:
        return False, "The image contains too much color to be a typical grayscale MRI scan."

    return True, "The image passed the basic MRI input checks."


def parse_arguments() -> argparse.Namespace:
    """Read the input MRI image path from the command line."""
    parser = argparse.ArgumentParser(
        description="Predict a brain tumor MRI class and visualize Grad-CAM."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to an MRI image, relative to the project root or an absolute path.",
    )
    return parser.parse_args()


def resolve_image_path(image_path: Path) -> Path:
    """Resolve a relative image path from the project root and validate it."""
    resolved_path = image_path if image_path.is_absolute() else PROJECT_ROOT / image_path
    resolved_path = resolved_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"MRI image not found: {resolved_path}")
    return resolved_path


def load_model(device: torch.device) -> nn.Module:
    """Build ResNet18 and load the project's saved model checkpoint."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}\n"
            "Train the model first so models/brain_tumor_resnet18.pth exists."
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load model checkpoint at {CHECKPOINT_PATH}: {error}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must be a dictionary.")

    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class order does not match the expected order. "
            f"Expected {CLASS_NAMES}, found {checkpoint_classes}."
        )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint does not contain a valid 'model_state_dict'."
        )

    # Do not request pretrained weights: the saved state dict provides them.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint parameters do not match the expected ResNet18 classifier: "
            f"{error}"
        ) from error

    return model.to(device).eval()


def load_image(image_path: Path) -> tuple[Image.Image, torch.Tensor]:
    """Open an MRI as RGB and return it with the normalized model input."""
    try:
        with Image.open(image_path) as source_image:
            original_image = source_image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not open MRI image {image_path}: {error}") from error

    image_tensor = IMAGE_TRANSFORM(original_image).unsqueeze(0)
    return original_image, image_tensor


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, float, np.ndarray]:
    """Return predicted class, confidence, and a normalized Grad-CAM heatmap."""
    image_tensor = image_tensor.to(device)

    # layer4[-1] is ResNet18's final residual block, producing 7x7 features
    # for a 224x224 input while retaining spatial information for Grad-CAM.
    target_layer = model.layer4[-1]
    captured_activations: list[torch.Tensor] = []
    captured_gradients: list[torch.Tensor] = []

    def save_activation(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        del _module, _inputs  # PyTorch supplies these hook arguments; only output is needed.
        captured_activations.append(output)
        output.register_hook(captured_gradients.append)

    hook_handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = probabilities[0].max(dim=0)

        # Backpropagate only the predicted class score to obtain its gradients.
        logits[0, predicted_index].backward()
    finally:
        hook_handle.remove()

    if not captured_activations or not captured_gradients:
        raise RuntimeError("Grad-CAM could not capture layer activations and gradients.")

    activations = captured_activations[0]
    gradients = captured_gradients[0]

    # Average each channel's gradients over the spatial dimensions, then
    # weight and combine the corresponding activation maps.
    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((channel_weights * activations).sum(dim=1, keepdim=True))
    heatmap = F.interpolate(
        heatmap,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    heatmap -= heatmap.min()
    maximum = heatmap.max()
    if maximum.item() > 0:
        heatmap /= maximum

    return (
        int(predicted_index.item()),
        float(confidence.item()),
        heatmap.detach().cpu().numpy(),
    )


def save_visualization(
    original_image: Image.Image,
    heatmap: np.ndarray,
    predicted_index: int,
    confidence: float,
    output_path: Path,
) -> None:
    """Save the original MRI, Grad-CAM heatmap, and overlay side by side."""
    # Resize the heatmap back to the original MRI dimensions for an aligned overlay.
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        original_image.size,
        resample=Image.Resampling.BILINEAR,
    )
    colorized_heatmap = plt.get_cmap("jet")(
        np.asarray(heatmap_image, dtype=np.float32) / 255.0
    )[..., :3]
    colorized_heatmap = Image.fromarray(np.uint8(colorized_heatmap * 255))
    overlay = Image.blend(original_image, colorized_heatmap, alpha=0.4)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original MRI")
    axes[1].imshow(colorized_heatmap)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {CLASS_NAMES[predicted_index]}\n"
        f"Confidence: {confidence:.2%}"
    )
    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    except OSError as error:
        raise OSError(f"Could not save Grad-CAM visualization to {output_path}: {error}") from error
    finally:
        plt.close(figure)


def main() -> None:
    """Run prediction, create Grad-CAM, and save the visualization."""
    arguments = parse_arguments()
    image_path = resolve_image_path(arguments.image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    original_image, image_tensor = load_image(image_path)
    predicted_index, confidence, heatmap = generate_gradcam(
        model,
        image_tensor,
        device,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{image_path.stem}_gradcam.png"
    save_visualization(
        original_image,
        heatmap,
        predicted_index,
        confidence,
        output_path,
    )

    print(f"Device: {device}")
    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Grad-CAM visualization saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error