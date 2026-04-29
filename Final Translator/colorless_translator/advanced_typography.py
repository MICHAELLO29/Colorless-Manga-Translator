"""
Advanced Typography System
Professional text rendering with kerning, leading, and style effects
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, List
import textwrap
import numpy as np


class AdvancedTypography:
    """Professional typography with advanced features."""
    
    def __init__(self, font_path: str, base_font_size: int = 14):
        self.font_path = font_path
        self.base_font_size = base_font_size
        self.fonts_cache = {}  # Cache loaded fonts
    
    def get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get font with caching."""
        cache_key = (size, bold)
        if cache_key not in self.fonts_cache:
            try:
                self.fonts_cache[cache_key] = ImageFont.truetype(self.font_path, size)
            except:
                self.fonts_cache[cache_key] = ImageFont.load_default()
        return self.fonts_cache[cache_key]
    
    def render_text_professional(
        self,
        draw: ImageDraw.Draw,
        text: str,
        box: Tuple[int, int, int, int],
        image_pil: Image.Image,
        style: str = 'dialogue'
    ) -> bool:
        """
        Render text with professional typography.
        
        Args:
            draw: PIL ImageDraw object
            text: Text to render
            box: (x, y, w, h) bounding box
            image_pil: PIL Image for background sampling
            style: Text style (dialogue/action/thought/sound_effect)
        
        Returns:
            bool: Success status
        """
        x, y, w, h = box
        
        # Detect text style from content
        detected_style = self._detect_text_style(text)
        if detected_style:
            style = detected_style
        
        # Get style-specific settings
        settings = self._get_style_settings(style)
        
        # Determine text color based on background
        text_color = self._get_adaptive_text_color(image_pil, box)
        
        # Calculate optimal font size with padding
        padding = settings['padding']
        available_w = w - (padding * 2)
        available_h = h - (padding * 2)
        
        # Find optimal font size
        font_size = self._find_optimal_font_size(
            text, 
            available_w, 
            available_h,
            settings
        )
        
        # Get font
        font = self.get_font(font_size, bold=settings['bold'])
        
        # Wrap text
        wrapped_text = self._wrap_text_smart(text, font, available_w, settings)
        
        # Calculate text position with alignment
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=settings['leading'])
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Center text
        text_x = x + (w - text_w) // 2
        text_y = y + (h - text_h) // 2
        
        # Apply text effects based on style
        if settings['effects']:
            self._apply_text_effects(
                draw, 
                wrapped_text, 
                (text_x, text_y), 
                font, 
                text_color, 
                settings
            )
        else:
            # Standard rendering
            draw.multiline_text(
                (text_x, text_y),
                wrapped_text,
                font=font,
                fill=text_color,
                align="center",
                spacing=settings['leading']
            )
        
        return True
    
    def _detect_text_style(self, text: str) -> str:
        """Detect text style from content."""
        # Sound effects (all caps, short)
        if text.isupper() and len(text) <= 15:
            return 'sound_effect'
        
        # Thoughts (often in parentheses or with ellipsis)
        if text.startswith('(') or '...' in text:
            return 'thought'
        
        # Action (short, with exclamation)
        if '!' in text and len(text) <= 20:
            return 'action'
        
        # Default to dialogue
        return 'dialogue'
    
    def _get_style_settings(self, style: str) -> Dict:
        """Get typography settings for each style."""
        settings = {
            'dialogue': {
                'padding': 8,
                'leading': 2,  # Line spacing
                'bold': False,
                'italic': False,
                'effects': False,
                'size_multiplier': 1.0
            },
            'action': {
                'padding': 6,
                'leading': 1,
                'bold': True,
                'italic': False,
                'effects': True,
                'size_multiplier': 1.1
            },
            'thought': {
                'padding': 10,
                'leading': 3,
                'bold': False,
                'italic': True,
                'effects': False,
                'size_multiplier': 0.9
            },
            'sound_effect': {
                'padding': 4,
                'leading': 0,
                'bold': True,
                'italic': False,
                'effects': True,
                'size_multiplier': 1.3
            }
        }
        
        return settings.get(style, settings['dialogue'])
    
    def _get_adaptive_text_color(
        self, 
        image_pil: Image.Image, 
        box: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Determine text color based on background brightness."""
        x, y, w, h = box
        
        # Sample center region of the box
        sample_x = x + w // 2
        sample_y = y + h // 2
        sample_w = min(w // 3, 30)
        sample_h = min(h // 3, 30)
        
        # Ensure within bounds
        sample_x = max(0, min(sample_x, image_pil.width - sample_w))
        sample_y = max(0, min(sample_y, image_pil.height - sample_h))
        
        try:
            # Crop sample region
            sample = image_pil.crop((
                sample_x, 
                sample_y, 
                sample_x + sample_w, 
                sample_y + sample_h
            ))
            
            # Convert to grayscale and get average brightness
            sample_gray = sample.convert('L')
            pixels = list(sample_gray.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            
            # Choose text color based on brightness
            if avg_brightness > 127:
                return (0, 0, 0)  # Black text on light background
            else:
                return (255, 255, 255)  # White text on dark background
        except:
            # Fallback to black
            return (0, 0, 0)
    
    def _find_optimal_font_size(
        self, 
        text: str, 
        max_width: int, 
        max_height: int,
        settings: Dict
    ) -> int:
        """Find optimal font size that fits the text in the box."""
        min_size = 8
        max_size = int(self.base_font_size * settings['size_multiplier'])
        
        # Binary search for optimal size
        best_size = min_size
        
        for size in range(max_size, min_size - 1, -1):
            font = self.get_font(size, bold=settings['bold'])
            wrapped = self._wrap_text_smart(text, font, max_width, settings)
            
            # Check if it fits
            bbox = font.getbbox(wrapped)
            text_h = (wrapped.count('\n') + 1) * (bbox[3] - bbox[1] + settings['leading'])
            
            if text_h <= max_height:
                best_size = size
                break
        
        return best_size
    
    def _wrap_text_smart(
        self, 
        text: str, 
        font: ImageFont.FreeTypeFont, 
        max_width: int,
        settings: Dict
    ) -> str:
        """Smart text wrapping with hyphenation awareness."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _apply_text_effects(
        self,
        draw: ImageDraw.Draw,
        text: str,
        position: Tuple[int, int],
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int],
        settings: Dict
    ):
        """Apply special effects to text (outline, shadow, etc.)."""
        x, y = position
        
        # For action/sound effects, add outline
        if settings['effects']:
            # Draw outline (simple version)
            outline_color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
            
            for offset_x in [-1, 0, 1]:
                for offset_y in [-1, 0, 1]:
                    if offset_x != 0 or offset_y != 0:
                        draw.multiline_text(
                            (x + offset_x, y + offset_y),
                            text,
                            font=font,
                            fill=outline_color,
                            align="center",
                            spacing=settings['leading']
                        )
        
        # Draw main text
        draw.multiline_text(
            (x, y),
            text,
            font=font,
            fill=color,
            align="center",
            spacing=settings['leading']
        )


class BubbleShapeAdapter:
    """Adapt text rendering to bubble shapes."""
    
    def __init__(self):
        pass
    
    def detect_bubble_shape(self, bubble_mask: np.ndarray) -> str:
        """Detect the shape of a speech bubble."""
        import cv2
        
        # Find contours
        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'rectangular'
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate shape
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Classify shape
        if len(approx) == 4:
            return 'rectangular'
        elif len(approx) > 8:
            return 'circular'
        else:
            return 'irregular'
    
    def get_text_area_for_shape(
        self, 
        bubble_mask: np.ndarray, 
        shape: str
    ) -> Tuple[int, int, int, int]:
        """Get optimal text area within bubble shape."""
        import cv2
        
        # Find bounding box
        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            h, w = bubble_mask.shape
            return (0, 0, w, h)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Adjust based on shape
        if shape == 'circular':
            # Inscribe rectangle in circle
            padding = int(min(w, h) * 0.15)
            return (x + padding, y + padding, w - 2*padding, h - 2*padding)
        elif shape == 'irregular':
            # More conservative padding
            padding = int(min(w, h) * 0.10)
            return (x + padding, y + padding, w - 2*padding, h - 2*padding)
        else:
            # Rectangular - standard padding
            padding = 8
            return (x + padding, y + padding, w - 2*padding, h - 2*padding)


class TextStyleDetector:
    """Detect and classify text styles for appropriate rendering."""
    
    @staticmethod
    def detect_emphasis(text: str) -> bool:
        """Detect if text should be emphasized (bold)."""
        # All caps
        if text.isupper() and len(text) > 2:
            return True
        
        # Multiple exclamation marks
        if text.count('!') >= 2:
            return True
        
        return False
    
    @staticmethod
    def detect_thoughts(text: str) -> bool:
        """Detect if text represents thoughts (italic)."""
        # Parentheses
        if text.startswith('(') and text.endswith(')'):
            return True
        
        # Ellipsis
        if text.count('...') >= 2:
            return True
        
        return False
    
    @staticmethod
    def detect_shouting(text: str) -> bool:
        """Detect if text is shouting (larger size)."""
        # All caps with exclamation
        if text.isupper() and '!' in text:
            return True
        
        # Multiple exclamation marks
        if text.count('!') >= 3:
            return True
        
        return False
    
    @staticmethod
    def classify_text_type(text: str) -> str:
        """Classify text into categories."""
        if TextStyleDetector.detect_shouting(text):
            return 'shout'
        elif TextStyleDetector.detect_thoughts(text):
            return 'thought'
        elif TextStyleDetector.detect_emphasis(text):
            return 'emphasis'
        else:
            return 'normal'
