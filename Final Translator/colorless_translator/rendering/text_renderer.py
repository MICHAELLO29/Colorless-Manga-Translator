"""Text rendering for translated manga pages."""

import re
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

from colorless_translator.core.exceptions import RenderingError


class TextRenderer:
    """Handles text rendering into speech bubbles."""
    
    def __init__(self, font_path: str, base_font_size: int = 14):
        self.font_path = font_path
        self.base_font_size = base_font_size
        self._font_cache = {}
    
    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font with caching."""
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(self.font_path, size)
            except Exception:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]
    
    def render_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        box: Tuple[int, int, int, int],
        image: Optional[Image.Image] = None,
    ):
        """
        Render text within a bounding box.
        
        Args:
            draw: PIL ImageDraw instance
            text: Text to render
            box: (x, y, w, h) bounding box
            image: Optional image for background sampling
        """
        x, y, w, h = box

        if text and re.search(r"[a-z]", text):
            text = text.upper()
        
        text_color = self._determine_text_color(image, box) if image else (0, 0, 0)
        padding = self._calculate_padding(w, h)
        
        if w <= padding * 2 + 5 or h <= padding * 2 + 5:
            padding = 1
        
        inner_w = w - padding * 2
        inner_h = h - padding * 2
        inner_x = x + padding
        inner_y = y + padding
        
        if inner_w <= 0 or inner_h <= 0:
            return
        
        fit_result = self._fit_text(text, inner_w, inner_h, w / h if h > 0 else 1)
        
        if fit_result is None:
            return
        
        font, wrapped, text_w, text_h, spacing = fit_result
        
        text_x = inner_x + (inner_w - text_w) / 2
        text_y = inner_y + (inner_h - text_h) / 2 - (font.size * 0.05)
        
        draw.multiline_text(
            (text_x, text_y),
            wrapped,
            font=font,
            fill=text_color,
            align="center",
            spacing=spacing,
        )
    
    def _determine_text_color(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int]:
        """Determine text color based on background brightness."""
        x, y, w, h = box
        
        try:
            sample_x = x + w // 2
            sample_y = y + h // 2
            sample_size = min(w, h) // 4
            
            crop_box = (
                max(0, sample_x - sample_size),
                max(0, sample_y - sample_size),
                min(image.width, sample_x + sample_size),
                min(image.height, sample_y + sample_size),
            )
            
            sample = image.crop(crop_box).convert("L")
            pixels = list(sample.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            
            if avg_brightness < 100:
                return (255, 255, 255)
            elif avg_brightness > 180:
                return (0, 0, 0)
            else:
                return (255, 255, 255) if avg_brightness < 140 else (0, 0, 0)
        except Exception:
            return (0, 0, 0)
    
    def _calculate_padding(self, w: int, h: int) -> int:
        """Calculate appropriate padding for box size."""
        padding = max(2, min(8, int(min(w, h) * 0.045)))
        if h > w * 2.2:
            padding = max(1, min(padding, 3))
        if w < 30 or h < 30:
            padding = max(1, int(min(w, h) * 0.03))
        return padding
    
    def _fit_text(
        self,
        text: str,
        max_w: int,
        max_h: int,
        aspect_ratio: float,
    ) -> Optional[Tuple]:
        """Find the largest font size where text fits within the box.

        Uses binary search: try a font size, wrap the text, check if it
        fits.  The largest fitting size wins.
        """
        normalized = " ".join(text.strip().split())
        if not normalized:
            return None

        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        def _measure_line(font: ImageFont.FreeTypeFont, s: str) -> float:
            try:
                return float(font.getlength(s))
            except Exception:
                bbox = font.getbbox(s)
                return float(bbox[2] - bbox[0])

        def _wrap(font: ImageFont.FreeTypeFont, s: str, width_px: int) -> list[str]:
            """Word-wrap text to width_px.  Never splits mid-word."""
            words = re.findall(r"\S+", s)
            if not words:
                return [""]

            lines: list[str] = []
            current = ""

            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if _measure_line(font, candidate) <= width_px:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    # If single word is wider than box, still keep it
                    current = word

            if current:
                lines.append(current)

            return lines

        def _try_size(size: int) -> Optional[Tuple]:
            """Try to fit text at a given font size. Returns fit tuple or None."""
            font = self.get_font(size)
            spacing = max(1, int(size * 0.20))

            # Use 85% of box to leave breathing room
            usable_w = int(max_w * 0.85)
            usable_h = int(max_h * 0.85)

            lines = _wrap(font, normalized, usable_w)

            # Reject if too many lines (unreadable)
            if len(lines) > 10:
                return None

            wrapped = "\n".join(lines)
            bbox = dummy_draw.multiline_textbbox(
                (0, 0), wrapped, font=font, spacing=spacing, align="center"
            )
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            if text_w > usable_w or text_h > usable_h:
                return None

            return (font, wrapped, text_w, text_h, spacing)

        # --- Binary search for the largest fitting font size ---
        min_size = 7
        max_size = min(18, max(min_size, int(min(max_w, max_h) * 0.28)))

        best_fit = None
        lo, hi = min_size, max_size

        while lo <= hi:
            mid = (lo + hi) // 2
            result = _try_size(mid)
            if result is not None:
                best_fit = result
                lo = mid + 1  # try larger
            else:
                hi = mid - 1  # too big, try smaller

        # Fallback: absolute minimum
        if best_fit is None:
            best_fit = _try_size(min_size)

        if best_fit is None:
            font = self.get_font(min_size)
            spacing = 1
            lines = _wrap(font, normalized, int(max_w * 0.95))
            wrapped = "\n".join(lines)
            bbox = dummy_draw.multiline_textbbox(
                (0, 0), wrapped, font=font, spacing=spacing, align="center"
            )
            best_fit = (font, wrapped, bbox[2] - bbox[0], bbox[3] - bbox[1], spacing)

        return best_fit
