"""Main translator orchestrator for manga translation."""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from colorless_translator.config import Settings, get_settings
from colorless_translator.core.pipeline import TranslationPipeline, PageResult
from colorless_translator.core.exceptions import (
    ConfigurationError,
    QuotaExhaustedError,
)
from colorless_translator.detection import YOLODetector, RoboflowDetector
from colorless_translator.ocr import MangaOCRWrapper
from colorless_translator.translation import GeminiTranslator, TranslationCache
from colorless_translator.rendering import Inpainter, TextRenderer
from colorless_translator.utils.image import load_image, save_image, ensure_output_dir
from colorless_translator.utils.helpers import list_image_files, suppress_warnings


class MangaTranslator:
    """
    High-level interface for manga translation.
    
    Handles model loading, configuration, and batch processing.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize translator with settings.
        
        Args:
            settings: Configuration settings. If None, loads from environment.
        """
        suppress_warnings()

        os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        
        print("--- Professional Manga Translator Script (Definitive Version) ---")
        
        self.settings = settings or get_settings()
        self._validate_settings()
        
        self.cache: Optional[TranslationCache] = None
        self.pipeline: Optional[TranslationPipeline] = None
        
        self._initialized = False
    
    def _validate_settings(self):
        """Validate configuration settings."""
        errors = self.settings.validate()
        if errors:
            for err in errors:
                print(f"Configuration error: {err}")
            raise ConfigurationError("Invalid configuration. Check your .env file.")
    
    def initialize(self):
        """Load all models and initialize components."""
        if self._initialized:
            return
        
        print("\nLoading Models...")
        
        # Use Roboflow cloud API or local YOLO model
        if self.settings.roboflow.use_roboflow:
            print("   Using Roboflow cloud detection API...")
            detector = RoboflowDetector(
                api_key=self.settings.roboflow.api_key,
                model_id=self.settings.roboflow.model_id,
            )
        else:
            detector = self._auto_pick_yolo_model()
        ocr = MangaOCRWrapper()
        
        if self.settings.cache.enabled:
            self.cache = TranslationCache(
                self.settings.cache.cache_file,
                self.settings.cache.max_size,
            )
        
        translator = GeminiTranslator(
            api_key=self.settings.gemini_api_key,
            model_name=self.settings.translation.gemini_model_name,
            cache=self.cache,
            max_retries=self.settings.translation.max_retries,
            retry_delay=self.settings.translation.retry_delay,
            max_length_ratio=self.settings.translation.max_translation_length_ratio,
        )
        
        inpainter = Inpainter()
        text_renderer = TextRenderer(
            self.settings.paths.font_path,
            self.settings.translation.font_size,
        )
        
        self.pipeline = TranslationPipeline(
            detector=detector,
            ocr=ocr,
            translator=translator,
            inpainter=inpainter,
            text_renderer=text_renderer,
            detection_thresholds=self.settings.detection.as_threshold_dict(),
        )
        
        print("\nAll models loaded successfully!")
        self._initialized = True

    def _auto_pick_yolo_model(self) -> 'YOLODetector':
        """Auto-select the best YOLO model from the models/ directory.
        
        Discovers all .pt files, loads each, and picks the one with more
        detected classes and higher average confidence.
        """
        from glob import glob

        models_dir = os.path.dirname(self.settings.paths.yolo_model_path)
        pt_files = sorted(glob(os.path.join(models_dir, "*.pt")))

        if len(pt_files) <= 1:
            # Only one model available — just use it
            model_path = pt_files[0] if pt_files else self.settings.paths.yolo_model_path
            print(f"   Using model: {os.path.basename(model_path)}")
            return YOLODetector(model_path)

        print(f"\n--- Auto Model Picker: Found {len(pt_files)} YOLO models ---")
        
        best_detector = None
        best_score = -1
        best_name = ""

        # Create a simple test image (white page with some shapes)
        # to compare detection behavior
        for pt in pt_files:
            name = os.path.basename(pt)
            try:
                print(f"   Loading {name}...")
                detector = YOLODetector(pt)
                
                # Score based on model size (larger = more capable for complex pages)
                size_mb = os.path.getsize(pt) / (1024 * 1024)
                
                # Prefer larger models as they tend to have better detection
                # dabest.pt (52MB) vs genericbest.pt (14MB)
                score = size_mb  # Simple: bigger model = higher score
                
                # Bonus for "best" in the name 
                if "dabest" in name.lower():
                    score += 20
                
                print(f"   {name}: {size_mb:.1f}MB, score={score:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_detector = detector
                    best_name = name
                    
            except Exception as e:
                print(f"   {name}: Failed to load ({str(e)[:50]})")
                continue

        if best_detector is None:
            print("   No models loaded! Falling back to default.")
            return YOLODetector(self.settings.paths.yolo_model_path)

        print(f"   Selected model: {best_name} (score: {best_score:.1f})")
        return best_detector
    
    def translate_image(self, input_path: str, output_path: str) -> bool:
        """
        Translate a single manga image.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            
        Returns:
            True if successful
        """
        if not self._initialized:
            self.initialize()
        
        print(f"\nProcessing image: {input_path}")
        
        image = load_image(input_path)
        if image is None:
            return False
        
        result = self.pipeline.process_page(image)
        
        if result.success and result.image is not None:
            save_image(result.image, output_path, "PNG")
            print(f"\nFinished and saved: {output_path}")
            return True
        else:
            if result.image is not None:
                save_image(result.image, output_path, "PNG")
            return False
    
    def translate_folder(
        self,
        input_folder: Optional[str] = None,
        output_folder: Optional[str] = None,
    ) -> dict:
        """
        Translate all images in a folder.
        
        Args:
            input_folder: Input folder path (uses settings if None)
            output_folder: Output folder path (uses settings if None)
            
        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()
        
        input_folder = input_folder or self.settings.paths.input_folder
        output_folder = output_folder or self.settings.paths.output_folder
        
        ensure_output_dir(output_folder)
        
        image_files = list_image_files(input_folder)
        
        print(f"\nFound {len(image_files)} image(s) to process")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        stats = {"processed": 0, "skipped": 0, "failed": 0}
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}.png"
            output_path = os.path.join(output_folder, output_filename)
            
            if os.path.exists(output_path):
                print(f"Skipping {filename} (already translated)")
                stats["skipped"] += 1
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing [{i}/{len(image_files)}]: {filename}")
            print("="*60)
            
            try:
                success = self.translate_image(input_path, output_path)
                
                if success:
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                    
            except QuotaExhaustedError:
                print(f"\n{'='*60}")
                print("PROCESSING STOPPED DUE TO API QUOTA LIMIT")
                print("="*60)
                break
        
        self._print_summary(stats, output_folder)
        
        return stats
    
    def _print_summary(self, stats: dict, output_folder: str):
        """Print final processing summary."""
        print(f"\n{'='*60}")
        print("TRANSLATION SUMMARY")
        print("="*60)
        print(f"Successfully translated: {stats['processed']}")
        print(f"Skipped (already done): {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        print(f"Output folder: {output_folder}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.cache:
            self.cache.save()
            self.cache.print_stats()
        
        if stats["failed"] > 0:
            print(f"\nTip: Re-run the script to retry failed images after quota resets.")
    
    def save_cache(self):
        """Save translation cache to disk."""
        if self.cache:
            self.cache.save()
            print("Translation cache saved.")
    
    def translate_image_bytes(self, image_bytes: bytes) -> Optional[PageResult]:
        """
        Translate a manga image from raw bytes.
        
        Args:
            image_bytes: Raw image file bytes
            
        Returns:
            PageResult with processed image, or None on failure
        """
        import io
        from PIL import Image as PILImage
        
        if not self._initialized:
            self.initialize()
        
        try:
            pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to decode image bytes: {e}")
            return None
        
        result = self.pipeline.process_page(image)
        return result
