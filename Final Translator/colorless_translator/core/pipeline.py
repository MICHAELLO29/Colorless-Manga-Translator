"""Translation pipeline for processing manga pages."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image, ImageDraw

from colorless_translator.core.exceptions import (
    QuotaExhaustedError,
    OCRError,
    RenderingError,
)
from colorless_translator.detection import YOLODetector, RoboflowDetector, RegionMerger, StrategyAnalyzer
from colorless_translator.ocr import MangaOCRWrapper
from colorless_translator.translation import GeminiTranslator, TranslationCache
from colorless_translator.rendering import Inpainter, TextRenderer
from colorless_translator.utils.helpers import has_japanese_text
from colorless_translator.utils.image import cv2_to_pil


@dataclass
class TextBlock:
    """Represents a detected text region with associated data."""
    box: Tuple[int, int, int, int]
    japanese_text: str
    english_text: str = ""
    text_mask: Optional[np.ndarray] = None
    class_name: str = ""
    confidence: float = 0.0
    quality_score: float = 0.0


@dataclass
class PageResult:
    """Result of processing a single page."""
    success: bool
    image: Optional[np.ndarray] = None
    blocks_processed: int = 0
    blocks_failed: int = 0
    strategy: str = "standard"
    error: Optional[str] = None


class TranslationPipeline:
    """Orchestrates the full translation pipeline for manga pages."""
    
    def __init__(
        self,
        detector: Union[YOLODetector, RoboflowDetector],
        ocr: MangaOCRWrapper,
        translator: GeminiTranslator,
        inpainter: Inpainter,
        text_renderer: TextRenderer,
        detection_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.detector = detector
        self.ocr = ocr
        self.translator = translator
        self.inpainter = inpainter
        self.text_renderer = text_renderer
        self.thresholds = detection_thresholds or {}
        
        self.merger = RegionMerger()
        self.strategy_analyzer = StrategyAnalyzer(detector)
    
    def process_page(
        self,
        image: np.ndarray,
        page_num: int = 1,
    ) -> PageResult:
        """
        Process a single manga page through the full pipeline.
        
        Args:
            image: BGR image as numpy array
            page_num: Page number for context
            
        Returns:
            PageResult with processed image and statistics
        """
        try:
            strategy, _, _ = self._analyze_strategy(image)
            
            detections = self._detect_text_regions(image)
            if not detections:
                print(" -> No text regions found.")
                return PageResult(success=True, image=image, strategy=strategy)
            
            blocks = self._extract_text_blocks(image, detections)
            if not blocks:
                print(" -> No valid text found.")
                return PageResult(success=True, image=image, strategy=strategy)
            
            blocks = self._translate_blocks(blocks, strategy)
            
            result_image = self._render_translations(image, blocks)
            
            failed = sum(1 for b in blocks if b.english_text == "[Translation Error]")
            
            return PageResult(
                success=True,
                image=result_image,
                blocks_processed=len(blocks),
                blocks_failed=failed,
                strategy=strategy,
            )
            
        except QuotaExhaustedError as e:
            print(f"\nAPI quota exhausted. Saving progress...")
            return PageResult(success=False, image=image, error=str(e))
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            return PageResult(success=False, image=image, error=str(e))
    
    def _analyze_strategy(self, image: np.ndarray) -> Tuple[str, Optional[Dict], Dict]:
        """Analyze page to determine translation strategy."""
        print("--- Pass 0: Strategy Debate & Adaptive Selection ---")
        return self.strategy_analyzer.analyze_page(image)
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple]:
        """Detect and filter text regions."""
        print("\n--- Pass 1: Detecting Text Regions ---")
        
        raw = self.detector.detect(image, self.thresholds)
        
        print(f"\n{'='*80}")
        print(f"RAW DETECTIONS: {len(raw)} REGIONS FOUND")
        print("="*80)
        
        for i, det in enumerate(raw):
            x, y, w, h, cls, conf = det
            print(f"  Raw #{i+1}: class={cls}, conf={conf:.3f}, size={w}x{h}, pos=({x},{y})")
        
        print(f"\nMERGING overlapping regions...")
        merged = self.merger.merge_overlapping_regions(raw)
        print(f"   Result: {len(raw)} -> {len(merged)} regions")
        
        print(f"\nFILTERING duplicate regions...")
        filtered = self.merger.sort_and_filter(merged)
        print(f"   Result: {len(merged)} -> {len(filtered)} regions")
        
        print(f"\nFINAL: {len(filtered)} regions to translate")
        print("="*80 + "\n")
        
        self._print_detection_stats(raw, merged)
        
        return filtered
    
    def _extract_text_blocks(
        self,
        image: np.ndarray,
        detections: List[Tuple],
    ) -> List[TextBlock]:
        """Extract text from detected regions using OCR."""
        print(f"\nProcessing {len(detections)} text regions...")
        print("="*80)
        
        blocks = []
        h_img, w_img = image.shape[:2]
        
        for i, det in enumerate(detections):
            x, y, w, h, class_name, conf = det
            print(f"\nProcessing Block #{i+1}: pos=({x},{y}), size={w}x{h}, conf={conf:.3f}")
            
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            if w <= 0 or h <= 0:
                print(f" -> Block #{i+1}: Invalid dimensions, skipping")
                continue
            
            roi = image[y:y+h, x:x+w]
            
            try:
                japanese_text, text_mask = self.ocr.extract_text_with_mask(roi)
            except OCRError as e:
                print(f" -> Block #{i+1}: OCR failed ({str(e)[:50]}), skipping")
                continue
            
            if not has_japanese_text(japanese_text):
                print(f" -> Block #{i+1} SKIPPED: No Japanese text (OCR: '{japanese_text}')")
                continue
            
            print(f" -> Block #{i+1} [{class_name}, conf={conf:.2f}]: '{japanese_text}'")
            
            blocks.append(TextBlock(
                box=(x, y, w, h),
                japanese_text=japanese_text,
                text_mask=text_mask,
                class_name=class_name,
                confidence=conf,
            ))
        
        return blocks
    
    def _translate_blocks(
        self,
        blocks: List[TextBlock],
        strategy: str,
    ) -> List[TextBlock]:
        """Translate all text blocks."""
        print(f"\n--- Translating Page ({strategy.upper()} strategy) ---")
        
        texts = [b.japanese_text for b in blocks]
        translations = self.translator.translate_batch(texts, strategy)
        
        while len(translations) < len(blocks):
            translations.append("[Translation Error]")
        
        for block, trans in zip(blocks, translations):
            block.english_text = trans
        
        return blocks
    
    def _render_translations(
        self,
        image: np.ndarray,
        blocks: List[TextBlock],
    ) -> np.ndarray:
        """Render translations onto the image.

        Strategy:
        1. Fill each detection bounding box with the sampled bg colour.
           This erases ALL Japanese text in one shot.
        2. Render English text in the available space.

        The detection model gives tight boxes around text regions inside
        speech bubbles, so filling the full box is correct and reliable.
        """
        print("\n--- Pass 2: Inpainting & Typesetting ---")

        result = image.copy()
        original = image  # clean reference for colour sampling

        render_boxes: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}

        for i, block in enumerate(blocks):
            preview = block.english_text[:50] + "..." if len(block.english_text) > 50 else block.english_text
            print(f' -> Cleaning Block #{i+1}: "{preview}"')

            x, y, w, h = block.box
            if w <= 0 or h <= 0:
                continue

            # -- 1. Sample the background colour from the ROI --------------
            roi = original[y:y+h, x:x+w].copy()
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
            fill_color = self._sample_fill_color(gray, roi)
            print(f"    box=({x},{y},{w},{h}), fill={fill_color}")

            # -- 2. Fill the ENTIRE bounding box with the bg colour --------
            #    This guarantees every Japanese text pixel is erased.
            result_roi = result[y:y+h, x:x+w]
            if result_roi.ndim == 3:
                result_roi[:, :] = fill_color
            else:
                result_roi[:, :] = fill_color[0]
            result[y:y+h, x:x+w] = result_roi

            # Store the fill box for later grouping
            render_boxes[block.box] = (x, y, w, h)

        # -- 3. Group adjacent blocks into merged render regions -----------
        #    Japanese vertical text columns are detected separately, but
        #    English text needs to be rendered horizontally across the
        #    full bubble width.  We merge boxes that overlap vertically
        #    and are horizontally close.
        merged_groups = self._group_adjacent_blocks(blocks)
        print(f"\n    Grouped {len(blocks)} blocks -> {len(merged_groups)} render groups")

        # -- 4. Render English text on the cleaned image -------------------
        pil_image = cv2_to_pil(result)
        draw = ImageDraw.Draw(pil_image)

        for group in merged_groups:
            # Compute bounding rect of the whole group
            xs = [b.box[0] for b in group]
            ys = [b.box[1] for b in group]
            x2s = [b.box[0] + b.box[2] for b in group]
            y2s = [b.box[1] + b.box[3] for b in group]
            gx = min(xs)
            gy = min(ys)
            gw = max(x2s) - gx
            gh = max(y2s) - gy

            # Expand the render area slightly beyond the tight detection
            # boxes to use more of the available white space in the bubble.
            # We already filled these areas with bg colour, so it's safe.
            img_h, img_w = result.shape[:2]
            expand_x = int(gw * 0.15)
            expand_y = int(gh * 0.08)
            gx = max(0, gx - expand_x)
            gy = max(0, gy - expand_y)
            gw = min(img_w - gx, gw + expand_x * 2)
            gh = min(img_h - gy, gh + expand_y * 2)

            # Concatenate translations in manga reading order (right-to-left)
            sorted_group = sorted(group, key=lambda b: -b.box[0])
            combined_text = " ".join(
                b.english_text for b in sorted_group
                if b.english_text and b.english_text != "[Translation Error]"
            )
            if not combined_text:
                continue

            # Margin so text doesn't touch edges
            margin = max(3, min(8, int(min(gw, gh) * 0.05)))
            rx = gx + margin
            ry = gy + margin
            rw = gw - margin * 2
            rh = gh - margin * 2

            if rw <= 5 or rh <= 5:
                continue

            preview = combined_text[:60] + "..." if len(combined_text) > 60 else combined_text
            print(f'    Rendering group ({len(group)} blocks, {rw}x{rh}): "{preview}"')

            try:
                self.text_renderer.render_text(draw, combined_text, (rx, ry, rw, rh), pil_image)
            except Exception as e:
                print(f"    Text drawing failed: {str(e)[:50]}")

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_fill_color(gray: np.ndarray, roi_bgr: np.ndarray) -> tuple:
        """Sample the background colour from an ROI for fill.

        For white/bright bubbles (most common in manga), takes the
        brightest pixels as the background.  For dark backgrounds,
        takes the darkest pixels.
        """
        avg = float(np.mean(gray))

        try:
            if avg > 120:
                # Bright region: sample the whitest pixels
                bright_mask = gray >= max(200, int(avg))
                if np.count_nonzero(bright_mask) < 10:
                    bright_mask = gray >= int(avg * 0.85)
            else:
                # Dark region: sample the darkest pixels
                bright_mask = gray <= min(60, int(avg))
                if np.count_nonzero(bright_mask) < 10:
                    bright_mask = gray <= int(avg * 1.2)

            if roi_bgr.ndim == 3 and np.count_nonzero(bright_mask) > 10:
                bg_b = int(np.median(roi_bgr[:, :, 0][bright_mask]))
                bg_g = int(np.median(roi_bgr[:, :, 1][bright_mask]))
                bg_r = int(np.median(roi_bgr[:, :, 2][bright_mask]))
                return (bg_b, bg_g, bg_r)
        except Exception:
            pass

        # Fallback
        if avg > 120:
            return (255, 255, 255)
        v = int(avg)
        return (v, v, v)

    @staticmethod
    def _group_adjacent_blocks(blocks: List) -> List[List]:
        """Group adjacent text blocks that belong to the same bubble.

        Japanese vertical text columns are detected as separate narrow
        boxes.  For horizontal English rendering we merge columns that
        are vertically overlapping and horizontally close into one
        wide render region.

        Returns a list of groups, where each group is a list of
        TextBlock objects.
        """
        if not blocks:
            return []

        used = set()
        groups: List[List] = []

        for i, block_a in enumerate(blocks):
            if i in used:
                continue

            group = [block_a]
            used.add(i)

            # Iteratively find neighbours of the growing group
            changed = True
            while changed:
                changed = False

                for j, block_b in enumerate(blocks):
                    if j in used:
                        continue
                    xb, yb, wb, hb = block_b.box
                    
                    is_close = False
                    for block_in_group in group:
                        xa, ya, wa, ha = block_in_group.box
                        
                        # 1. Vertical overlap check
                        y_overlap = min(ya + ha, yb + hb) - max(ya, yb)
                        if y_overlap < min(ha, hb) * 0.25:
                            continue
                            
                        # 2. Similar width check (same font size)
                        # Columns in the same bubble generally have similar widths
                        if max(wa, wb) > min(wa, wb) * 1.8:
                            continue
                            
                        # 3. Horizontal proximity check
                        # Gap should be small compared to the text column width
                        x_gap = max(0, max(xa, xb) - min(xa + wa, xb + wb))
                        if x_gap > min(wa, wb) * 1.2:
                            continue
                            
                        # 4. Top-alignment check
                        # Columns in the same bubble usually start near each other vertically
                        if abs(ya - yb) > max(ha, hb) * 0.5:
                            continue
                            
                        is_close = True
                        break
                        
                    if is_close:
                        group.append(block_b)
                        used.add(j)
                        changed = True

            groups.append(group)

        return groups

    def _print_detection_stats(self, raw: List, merged: List):
        """Print detection statistics."""
        print("\nDetection Statistics:")
        
        class_counts = {}
        for det in raw:
            cls = det[4]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("  Raw detections:")
        for cls, count in sorted(class_counts.items()):
            print(f"    - {cls}: {count}")
        
        merged_counts = {}
        for reg in merged:
            reg_type = reg[4]
            merged_counts[reg_type] = merged_counts.get(reg_type, 0) + 1
        
        print(f"  Merged regions: {len(merged)}")
        for reg_type, count in sorted(merged_counts.items()):
            print(f"    - {reg_type}: {count}")
