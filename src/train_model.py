"""Train and evaluate a ResNet18 brain MRI classifier."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# Resolve paths from this file so the script works from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "dataset" / "Training"
TESTING_DIR = PROJECT_ROOT / "dataset" / "Testing"
MODEL_PATH = PROJECT_ROOT / "models" / "brain_tumor_resnet18.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42

# ResNet18 was pretrained on ImageNet, so use its matching normalization.
NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
EVALUATION_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def check_dataset_folders() -> None:
    """Check that both dataset splits contain all expected class folders."""
    for split_dir in (TRAINING_DIR, TESTING_DIR):
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {split_dir}\n"
                "Expected dataset/Training and dataset/Testing beside the src folder."
            )

        missing_classes = [
            class_name
            for class_name in CLASS_NAMES
            if not (split_dir / class_name).is_dir()
        ]
        if missing_classes:
            raise FileNotFoundError(
                f"Missing class folders in {split_dir}: "
                f"{', '.join(missing_classes)}"
            )


def create_datasets() -> tuple[
    datasets.ImageFolder,
    datasets.ImageFolder,
    datasets.ImageFolder,
]:
    """Load image folders and verify their labels match the required classes."""
    training_dataset = datasets.ImageFolder(
        root=str(TRAINING_DIR),
        transform=TRAIN_TRANSFORMS,
    )
    validation_dataset = datasets.ImageFolder(
        root=str(TRAINING_DIR),
        transform=EVALUATION_TRANSFORMS,
    )
    testing_dataset = datasets.ImageFolder(
        root=str(TESTING_DIR),
        transform=EVALUATION_TRANSFORMS,
    )

    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    for split_name, dataset in (
        ("training", training_dataset),
        ("validation", validation_dataset),
        ("testing", testing_dataset),
    ):
        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"The {split_name} dataset classes are {dataset.class_to_idx}; "
                f"expected {expected_mapping}."
            )
        if not dataset.samples:
            raise ValueError(f"No supported images were found in the {split_name} dataset.")
        class_counts = {name: 0 for name in CLASS_NAMES}
        for _, label in dataset.samples:
            class_counts[CLASS_NAMES[label]] += 1
        empty_classes = [name for name, count in class_counts.items() if count == 0]
        if empty_classes:
            raise ValueError(
                f"The {split_name} dataset has no images for: "
                f"{', '.join(empty_classes)}."
            )

    # Both training-folder datasets must have the same ordering for the split indices.
    if training_dataset.samples != validation_dataset.samples:
        raise RuntimeError("Training and validation image lists do not match.")

    return training_dataset, validation_dataset, testing_dataset


def make_data_loaders(
    training_dataset: datasets.ImageFolder,
    validation_dataset: datasets.ImageFolder,
    testing_dataset: datasets.ImageFolder,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible, stratified training and validation subsets."""
    labels = [label for _, label in training_dataset.samples]
    class_counts = [labels.count(index) for index in range(len(CLASS_NAMES))]
    if min(class_counts) < 2:
        raise ValueError(
            "Each training class needs at least two images to create a stratified "
            f"training/validation split. Found counts: {dict(zip(CLASS_NAMES, class_counts))}"
        )

    validation_size = max(
        len(CLASS_NAMES),
        round(len(training_dataset) * VALIDATION_FRACTION),
    )
    training_size = len(training_dataset) - validation_size
    if training_size < len(CLASS_NAMES):
        raise ValueError(
            "Not enough training images for the requested validation split. "
            "Add more images to dataset/Training."
        )

    training_indices, validation_indices = train_test_split(
        list(range(len(training_dataset))),
        test_size=validation_size,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Zero workers avoids multiprocessing issues when running on Windows.
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(
        Subset(training_dataset, training_indices),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        shuffle=False,
        **loader_options,
    )
    testing_loader = DataLoader(
        testing_dataset,
        shuffle=False,
        **loader_options,
    )
    return training_loader, validation_loader, testing_loader


def build_model(device: torch.device) -> nn.Module:
    """Load pretrained ResNet18 weights and replace its final classification layer."""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

    return total_loss / total_images, correct_predictions / total_images


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Evaluate the model and return average loss, true labels, and predictions."""
    model.eval()
    total_loss = 0.0
    total_images = 0
    true_labels: list[int] = []
    predictions: list[int] = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    return total_loss / total_images, true_labels, predictions


def save_confusion_matrix(
    true_labels: Sequence[int],
    predictions: Sequence[int],
) -> None:
    """Save a labeled confusion matrix image in the results directory."""
    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title("Brain Tumor MRI Confusion Matrix")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    """Train the classifier, keep its best validation checkpoint, and test it."""
    check_dataset_folders()
    training_dataset, validation_dataset, testing_dataset = create_datasets()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training images: {len(training_dataset)}")
    print(f"Testing images: {len(testing_dataset)}")
    print(
        "Loading pretrained ResNet18 weights. "
        "The first run may download these model weights if they are not cached."
    )

    training_loader, validation_loader, testing_loader = make_data_loaders(
        training_dataset,
        validation_dataset,
        testing_dataset,
        device,
    )
    model = build_model(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        training_loss, training_accuracy = train_one_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
            device,
        )
        validation_loss, validation_labels, validation_predictions = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )
        validation_accuracy = accuracy_score(
            validation_labels,
            validation_predictions,
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Training Loss: {training_loss:.4f} | "
            f"Training Accuracy: {training_accuracy:.2%} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%}"
        )

        # Use validation results (not test results) to select the best checkpoint.
        if (
            validation_accuracy > best_validation_accuracy
            or (
                validation_accuracy == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
        ):
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                },
                MODEL_PATH,
            )

    # Reload the best checkpoint before measuring final test-set performance.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_labels, test_predictions = evaluate(
        model,
        testing_loader,
        loss_function,
        device,
    )
    test_accuracy = accuracy_score(test_labels, test_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="weighted",
        zero_division=0,
    )

    print("\nTest set results (weighted precision, recall, and F1-score):")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1_score:.4f}")

    save_confusion_matrix(test_labels, test_predictions)
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()