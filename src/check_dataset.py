from pathlib import Path

# Project dataset folder
dataset_path = Path("dataset")

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

print("BRAIN TUMOR DATASET CHECK")
print("=" * 40)

for split in ["Training", "Testing"]:
    split_path = dataset_path / split

    print(f"\n{split.upper()}")

    if not split_path.exists():
        print(f"ERROR: {split} folder not found!")
        continue

    for class_folder in sorted(split_path.iterdir()):
        if class_folder.is_dir():
            count = sum(
                1 for file in class_folder.rglob("*")
                if file.suffix.lower() in extensions
            )
            print(f"{class_folder.name}: {count} images")