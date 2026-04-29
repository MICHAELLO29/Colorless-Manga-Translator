"""Core module containing the main translation orchestration logic."""

from colorless_translator.core.translator import MangaTranslator
from colorless_translator.core.pipeline import TranslationPipeline
from colorless_translator.core.exceptions import (
    TranslationError,
    QuotaExhaustedError,
    DetectionError,
    OCRError,
    RenderingError,
)

__all__ = [
    "MangaTranslator",
    "TranslationPipeline",
    "TranslationError",
    "QuotaExhaustedError",
    "DetectionError",
    "OCRError",
    "RenderingError",
]
