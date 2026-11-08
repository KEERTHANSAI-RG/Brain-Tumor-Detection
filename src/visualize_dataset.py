import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

dataset_path = Path("dataset")
classes = ["glioma", "meningioma", "notumor", "pituitary"]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for i, class_name in enumerate(classes):
    folder = dataset_path / "Training" / class_name

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    image_path = random.choice(images)

    image = Image.open(image_path)

    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(class_name.capitalize())
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("results/sample_mri_images.png")
plt.show()