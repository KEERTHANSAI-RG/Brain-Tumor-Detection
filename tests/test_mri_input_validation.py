"""Tests for the conservative MRI input checks and Streamlit upload gate."""

import io
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import CLASS_NAMES, validate_mri_image  # noqa: E402


def make_ordinary_color_photograph() -> Image.Image:
    """Create a colorful landscape-like image without an external test asset."""
    image = Image.new("RGB", (256, 256), color=(91, 174, 231))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 150, 256, 256), fill=(68, 151, 66))
    drawing.ellipse((25, 30, 78, 82), fill=(252, 221, 80))
    drawing.polygon(
        [(22, 185), (62, 100), (104, 185)],
        fill=(30, 112, 52),
    )
    drawing.polygon(
        [(150, 190), (202, 90), (250, 190)],
        fill=(40, 125, 53),
    )
    # A brown, dog-shaped figure makes the test input photo-like and colorful.
    drawing.ellipse((91, 161, 161, 217), fill=(150, 82, 45))
    drawing.ellipse((139, 158, 177, 190), fill=(166, 94, 54))
    drawing.ellipse((162, 162, 180, 177), fill=(32, 27, 24))
    drawing.rectangle((102, 202, 114, 238), fill=(102, 56, 34))
    drawing.rectangle((145, 202, 157, 238), fill=(102, 56, 34))
    return image


class MRIInputValidationTests(unittest.TestCase):
    def test_testing_scans_from_each_class_pass_validation(self) -> None:
        for class_name in CLASS_NAMES:
            with self.subTest(class_name=class_name):
                class_directory = PROJECT_ROOT / "dataset" / "Testing" / class_name
                image_paths = [
                    path
                    for path in class_directory.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ]
                self.assertTrue(image_paths, f"No test scan found for {class_name}")
                with Image.open(image_paths[0]) as source:
                    image = source.convert("RGB")

                is_valid, reason = validate_mri_image(image)

                self.assertTrue(is_valid, reason)

    def test_ordinary_color_photograph_is_rejected(self) -> None:
        is_valid, reason = validate_mri_image(make_ordinary_color_photograph())

        self.assertFalse(is_valid)
        self.assertIn("color", reason.lower())

    def test_photo_upload_is_rejected_before_analyze_button_is_shown(self) -> None:
        photo = make_ordinary_color_photograph()
        image_bytes = io.BytesIO()
        photo.save(image_bytes, format="JPEG")

        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=120,
        ).run()
        app.file_uploader[0].upload(
            "ordinary_photo.jpg",
            image_bytes.getvalue(),
            "image/jpeg",
        ).run()

        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertTrue(
            any(
                item.value == "Invalid input. Please upload a valid brain MRI scan."
                for item in app.error
            )
        )
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.metric), 0)

    def test_unsupported_image_format_is_rejected(self) -> None:
        image_bytes = io.BytesIO()
        make_ordinary_color_photograph().save(image_bytes, format="TIFF")

        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=120,
        ).run()
        # A misleading extension must not bypass validation of the actual file format.
        app.file_uploader[0].upload(
            "scan.jpg",
            image_bytes.getvalue(),
            "image/jpeg",
        ).run()

        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertTrue(
            any(
                item.value == "Invalid input. Please upload a valid brain MRI scan."
                for item in app.error
            )
        )
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
