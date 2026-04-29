"""Common helper functions."""

import re
import os
from pathlib import Path
from typing import Optional


def has_japanese_text(text: str) -> bool:
    """Check if text contains Japanese characters."""
    if not text:
        return False
    return bool(re.search(r'[ぁ-んァ-ン一-龯]', text))


def extract_page_number(filename: str) -> int:
    """Extract page number from filename."""
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 1


def get_output_filename(input_path: str, output_ext: str = ".png") -> str:
    """Generate output filename from input path."""
    stem = Path(input_path).stem
    return f"{stem}{output_ext}"


def list_image_files(directory: str) -> list[str]:
    """List all image files in directory."""
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])


def suppress_warnings():
    """Suppress TensorFlow and other warnings."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    import warnings
    warnings.filterwarnings('ignore')
