"""
Colorless Manga Translator
A professional manga translation toolkit with YOLO detection, OCR, and AI translation.
"""

__version__ = "2.0.0"
__author__ = "Colorless Manga Translator Team"

from colorless_translator.core.translator import MangaTranslator
from colorless_translator.core.exceptions import (
    TranslationError,
    QuotaExhaustedError,
    DetectionError,
    OCRError,
    RenderingError,
)

__all__ = [
    "MangaTranslator",
    "TranslationError",
    "QuotaExhaustedError",
    "DetectionError",
    "OCRError",
    "RenderingError",
]
