"""MangaOCR wrapper for Japanese text extraction."""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from colorless_translator.core.exceptions import OCRError, ModelLoadError


class MangaOCRWrapper:
    """Wrapper for MangaOCR with text mask generation."""
    
    def __init__(self):
        self.mocr = None
        self._load_model()
    
    def _load_model(self):
        """Load MangaOCR model."""
        try:
            print("Loading MangaOCR (this may take 20-30 seconds)...")
            from manga_ocr import MangaOcr
            self.mocr = MangaOcr()
            print("MangaOCR loaded")
        except Exception as e:
            raise ModelLoadError(f"Failed to load MangaOCR: {e}") from e
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract Japanese text from image region.
        
        Args:
            image: BGR image region as numpy array
            
        Returns:
            Extracted Japanese text
        """
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return self.mocr(pil_image)
        except Exception as e:
            raise OCRError(f"Text extraction failed: {e}") from e
    
    def extract_text_with_mask(
        self, 
        image: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        """
        Extract text and generate text mask for inpainting.
        
        Args:
            image: BGR image region as numpy array
            
        Returns:
            Tuple of (extracted_text, text_mask)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        is_dark = avg_brightness < 80
        is_gray = 80 <= avg_brightness < 150
        
        if is_dark or is_gray:
            text_mask = self._create_light_text_mask(gray)
        else:
            text_mask = self._create_dark_text_mask(gray)

        mask_area = int(np.count_nonzero(text_mask))
        roi_area = int(text_mask.shape[0] * text_mask.shape[1])
        coverage = (mask_area / roi_area) if roi_area > 0 else 0.0
        if coverage > 0.45:
            kernel = np.ones((3, 3), np.uint8)
            text_mask = cv2.erode(text_mask, kernel, iterations=1)
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        japanese_text = self.mocr(pil_image)
        
        return japanese_text, text_mask
    
    def _create_light_text_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create mask for light text on dark background."""
        _, mask1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        mask3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, -5
        )
        mask4 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, -3
        )
        
        combined = cv2.bitwise_or(mask1, mask2)
        combined = cv2.bitwise_or(combined, mask3)
        combined = cv2.bitwise_or(combined, mask4)

        tophat_kernel = np.ones((7, 7), np.uint8)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
        _, mask5 = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(combined, mask5)

        opening_kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, opening_kernel)
        
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(combined, kernel, iterations=2)
    
    def _create_dark_text_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create mask for dark text on light background."""
        mask1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        mask2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )
        _, mask3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        combined = cv2.bitwise_or(mask1, mask2)
        combined = cv2.bitwise_or(combined, mask3)

        blackhat_kernel = np.ones((7, 7), np.uint8)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        _, mask4 = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(combined, mask4)

        opening_kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, opening_kernel)
        
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(combined, kernel, iterations=2)
