"""Image loading and saving utilities."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        path: Path to image file
        
    Returns:
        BGR image as numpy array or None if loading fails
    """
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
    return image


def save_image(image: np.ndarray, path: str, format: str = "PNG") -> bool:
    """
    Save image to file.
    
    Args:
        image: BGR numpy array or PIL Image
        path: Output path
        format: Image format
        
    Returns:
        True if successful
    """
    try:
        if isinstance(image, np.ndarray):
            cv2.imwrite(path, image)
        elif isinstance(image, Image.Image):
            image.save(path, format)
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR image."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def ensure_output_dir(path: str):
    """Create output directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
