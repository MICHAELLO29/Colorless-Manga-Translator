"""Custom exceptions for the manga translation pipeline."""


class TranslationError(Exception):
    """Base exception for translation errors."""
    pass


class QuotaExhaustedError(TranslationError):
    """Raised when API quota is exhausted."""
    pass


class DetectionError(TranslationError):
    """Raised when text detection fails."""
    pass


class OCRError(TranslationError):
    """Raised when OCR processing fails."""
    pass


class RenderingError(TranslationError):
    """Raised when text rendering fails."""
    pass


class ConfigurationError(TranslationError):
    """Raised when configuration is invalid or missing."""
    pass


class ModelLoadError(TranslationError):
    """Raised when model loading fails."""
    pass
