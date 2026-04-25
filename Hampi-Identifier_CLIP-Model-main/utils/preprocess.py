"""
preprocess.py — Image preprocessing utilities for Hampi monument classifier.
"""

from PIL import Image, ImageOps, ImageFilter
import io
import base64


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_INPUT_SIZE = (224, 224)   # CLIP default
MAX_UPLOAD_MB = 10
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Load a PIL Image from a Streamlit UploadedFile object.
    Validates size and format; raises ValueError on invalid input.
    """
    # Size guard
    uploaded_file.seek(0, 2)  # seek to end
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). "
            f"Please upload an image under {MAX_UPLOAD_MB} MB."
        )

    image = Image.open(uploaded_file)
    fmt = image.format or "UNKNOWN"
    if fmt.upper() not in SUPPORTED_FORMATS and fmt != "UNKNOWN":
        raise ValueError(
            f"Unsupported format: {fmt}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return convert_to_rgb(image)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB, handling RGBA transparency gracefully."""
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def prepare_for_clip(image: Image.Image) -> Image.Image:
    """
    Prepare a PIL image for CLIP inference:
    - Auto-orient via EXIF
    - Convert to RGB
    Returns the processed PIL image (CLIP processor handles final resize/norm).
    """
    image = ImageOps.exif_transpose(image)  # fix phone rotation
    image = convert_to_rgb(image)
    return image


def resize_for_display(
    image: Image.Image,
    max_width: int = 600,
    max_height: int = 450,
) -> Image.Image:
    """Resize image proportionally for UI display while preserving aspect ratio."""
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    return image


def image_to_bytes(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> bytes:
    """Encode PIL image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def validate_image_quality(image: Image.Image) -> dict:
    """
    Run basic quality checks on the uploaded image.
    Returns a dict with 'ok' (bool) and 'warnings' (list[str]).
    """
    warnings = []
    w, h = image.size

    if w < 100 or h < 100:
        warnings.append("⚠️ Image resolution is very low — prediction may be inaccurate.")

    if w > 4000 or h > 4000:
        warnings.append("ℹ️ Large image detected — will be scaled internally for inference.")

    # Check if image is excessively dark or bright (simple mean pixel check)
    import numpy as np
    arr = np.array(image.convert("RGB"))
    mean_brightness = arr.mean()
    if mean_brightness < 20:
        warnings.append("⚠️ Image appears very dark — ensure the monument is clearly visible.")
    elif mean_brightness > 240:
        warnings.append("⚠️ Image appears overexposed — prediction may be affected.")

    return {"ok": len(warnings) == 0, "warnings": warnings}
