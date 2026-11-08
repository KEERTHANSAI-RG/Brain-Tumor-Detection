"""Streamlit interface for the trained brain tumor MRI classifier."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import matplotlib

# The application displays generated images in Streamlit rather than opening windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Make the src package importable when Streamlit is started from the project root.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import (  # noqa: E402
    CLASS_NAMES,
    CHECKPOINT_PATH,
    IMAGE_TRANSFORM,
    generate_gradcam,
    load_model,
    validate_mri_image,
)


SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP"}

st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the trained ResNet18 model...")
def get_model() -> tuple[torch.nn.Module, torch.device]:
    """Load and cache the existing classifier for this Streamlit session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(device), device


def decode_uploaded_image(image_bytes: bytes) -> Image.Image:
    """Validate an uploaded image and return a detached RGB copy."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image_format = uploaded_image.format
            rgb_image = uploaded_image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "The uploaded file could not be read as a valid image. "
            "Please choose a JPG, PNG, or BMP MRI image."
        ) from error

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ValueError(f"Unsupported image format. Please upload a {supported} image.")
    return rgb_image


def create_gradcam_images(
    original_image: Image.Image,
    heatmap: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    """Convert a Grad-CAM array to a color heatmap and MRI overlay."""
    grayscale_heatmap = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    ).resize(
        original_image.size,
        resample=Image.Resampling.BILINEAR,
    )
    color_values = plt.get_cmap("jet")(
        np.asarray(grayscale_heatmap, dtype=np.float32) / 255.0
    )[..., :3]
    colorized_heatmap = Image.fromarray(np.uint8(color_values * 255))
    overlay = Image.blend(original_image, colorized_heatmap, alpha=0.4)
    return colorized_heatmap, overlay


def main() -> None:
    """Render the image upload, prediction, and Grad-CAM explanation interface."""
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 1180px; padding-top: 2rem;}
        .hero {
            padding: 1.5rem 1.8rem; border-radius: 16px;
            background: linear-gradient(120deg, #102a43, #176b87);
            color: white; margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0 0 .35rem 0; color: white;}
        .hero p {margin: 0; color: #e6f4f7;}
        </style>
        <div class="hero">
          <h1>Brain Tumor MRI Classifier</h1>
          <p>Upload an MRI image to explore a model prediction and its Grad-CAM explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "**Educational project only.** This application is not a medical diagnostic "
        "tool and must not be used to diagnose, treat, or rule out any condition. "
        "Please consult a qualified medical professional."
    )

    with st.sidebar:
        st.header("About this demo")
        st.write(
            "This app uses a trained ResNet18 model and Grad-CAM to highlight image "
            "regions that influenced a prediction."
        )
        st.markdown("**Possible classes**")
        for class_name in CLASS_NAMES:
            st.markdown(f"- {class_name}")
        st.caption(f"Checkpoint: `{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}`")
        st.caption("Images are processed locally by this application.")

    upload_column, info_column = st.columns([1.25, 0.75], gap="large")
    with upload_column:
        uploaded_file = st.file_uploader(
            "Upload an MRI image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, PNG, and BMP.",
        )

    if uploaded_file is None:
        with info_column:
            st.subheader("How to use")
            st.markdown(
                "1. Choose an MRI image file.\n"
                "2. Review the image preview.\n"
                "3. Select **Analyze MRI** to see a prediction and Grad-CAM."
            )
        st.info("Upload an image to get started.")
        return

    image_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("image_key") != image_key:
        st.session_state["image_key"] = image_key
        st.session_state.pop("analysis", None)

    try:
        original_image = decode_uploaded_image(image_bytes)
    except ValueError as error:
        st.error("Invalid input. Please upload a valid brain MRI scan.")
        st.caption(str(error))
        return

    is_valid_mri, validation_message = validate_mri_image(original_image)
    if not is_valid_mri:
        st.error("Invalid input. Please upload a valid brain MRI scan.")
        st.caption(validation_message)
        return

    with upload_column:
        st.image(
            original_image,
            caption=f"Uploaded MRI: {uploaded_file.name}",
            width="stretch",
        )
    with info_column:
        st.subheader("Ready to analyze")
        st.write(
            "The model resizes the image to 224 × 224 pixels and applies the same "
            "normalization used during training."
        )
        st.caption("The prediction is an estimate from this educational model.")

    if st.button("Analyze MRI", type="primary", width="stretch"):
        st.session_state.pop("analysis", None)
        try:
            with st.spinner("Generating prediction and Grad-CAM..."):
                model, device = get_model()
                image_tensor = IMAGE_TRANSFORM(original_image).unsqueeze(0)
                predicted_index, confidence, heatmap = generate_gradcam(
                    model,
                    image_tensor,
                    device,
                )
                colorized_heatmap, overlay = create_gradcam_images(
                    original_image,
                    heatmap,
                )
                st.session_state["analysis"] = {
                    "class_name": CLASS_NAMES[predicted_index],
                    "confidence": confidence,
                    "heatmap": colorized_heatmap,
                    "overlay": overlay,
                    "device": str(device),
                }
        except (FileNotFoundError, ValueError, RuntimeError, OSError, IndexError) as error:
            st.error(f"Could not analyze this image: {error}")
            st.caption(
                "Check that the trained checkpoint exists and that the uploaded file "
                "is a valid MRI image."
            )

    analysis = st.session_state.get("analysis")
    if analysis is not None:
        st.divider()
        st.subheader("Prediction")
        result_column, confidence_column = st.columns(2)
        result_column.metric("Predicted class", analysis["class_name"])
        confidence_column.metric("Confidence", f"{analysis['confidence']:.2%}")
        st.progress(analysis["confidence"], text="Model confidence")

        heatmap_column, overlay_column = st.columns(2, gap="large")
        with heatmap_column:
            st.markdown("#### Grad-CAM heatmap")
            st.image(
                analysis["heatmap"],
                caption="Warmer colors indicate regions with greater influence.",
                width="stretch",
            )
        with overlay_column:
            st.markdown("#### Heatmap overlay")
            st.image(
                analysis["overlay"],
                caption="Grad-CAM heatmap blended with the uploaded MRI.",
                width="stretch",
            )
        st.caption(f"Inference device: {analysis['device']}")
        st.info(
            "Grad-CAM visualizes regions that influenced this model's output. "
            "It does not confirm the presence or absence of a tumor."
        )


if __name__ == "__main__":
    main()