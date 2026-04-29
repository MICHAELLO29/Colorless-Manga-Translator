"""Image inpainting for text removal."""

import numpy as np
import cv2

from colorless_translator.core.exceptions import RenderingError


def create_precise_bubble_mask(bubble_roi: np.ndarray) -> np.ndarray | None:
    if bubble_roi is None or bubble_roi.size == 0:
        return None

    gray = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY) if bubble_roi.ndim == 3 else bubble_roi
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape[:2]

    p80 = float(np.percentile(gray, 80))
    thresh = int(max(145, min(245, p80)))
    _, white = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, open_kernel, iterations=1)

    seed = (w // 2, h // 2)
    if white[seed[1], seed[0]] == 0:
        found = False
        max_r = max(5, min(h, w) // 6)
        for r in range(1, max_r):
            x0 = max(0, seed[0] - r)
            x1 = min(w - 1, seed[0] + r)
            y0 = max(0, seed[1] - r)
            y1 = min(h - 1, seed[1] + r)
            window = white[y0:y1 + 1, x0:x1 + 1]
            ys, xs = np.where(window > 0)
            if ys.size > 0:
                seed = (int(x0 + xs[0]), int(y0 + ys[0]))
                found = True
                break
        if not found:
            return None

    ff = white.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=seed, newVal=128)
    mask = np.where(ff == 128, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    mask = filled

    coverage = float(np.count_nonzero(mask)) / float(h * w) if h * w > 0 else 0.0
    if coverage < 0.02:
        return None

    if coverage > 0.88:
        border = 2
        top = gray[:border, :]
        bottom = gray[-border:, :]
        left = gray[:, :border]
        right = gray[:, -border:]
        edge = np.concatenate([
            top.reshape(-1),
            bottom.reshape(-1),
            left.reshape(-1),
            right.reshape(-1),
        ])
        dark_ratio = float(np.mean(edge < 160)) if edge.size > 0 else 0.0
        if dark_ratio < 0.02:
            return None

    erode_k = max(3, min(31, (int(min(h, w) * 0.03) | 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


class Inpainter:
    """Handles text removal via inpainting."""
    
    def inpaint_region(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        Inpaint a region to remove text.
        
        Args:
            image: Full BGR image
            mask: Text mask for the region
            x, y, w, h: Region coordinates
            
        Returns:
            Image with region inpainted
        """
        result = image.copy()
        
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            raise RenderingError(f"Region out of bounds: ({x}, {y}, {w}, {h})")
        
        roi = result[y:y+h, x:x+w]
        
        if np.sum(mask) == 0:
            return result
        
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        healed = self._apply_inpainting(roi, binary_mask, avg_brightness)
        result[y:y+h, x:x+w] = healed
        
        return result
    
    def _apply_inpainting(
        self,
        roi: np.ndarray,
        mask: np.ndarray,
        brightness: float,
    ) -> np.ndarray:
        """Apply appropriate inpainting based on background type."""
        is_dark = brightness < 80

        mask_area = int(np.count_nonzero(mask))
        roi_area = int(mask.shape[0] * mask.shape[1])
        coverage = (mask_area / roi_area) if roi_area > 0 else 0.0

        working = mask
        if coverage > 0.35:
            kernel = np.ones((3, 3), np.uint8)
            working = cv2.erode(working, kernel, iterations=1)
        
        # Dilate mask slightly so inpainting covers text edges cleanly
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(working, kernel, iterations=2)
        
        # Use Navier-Stokes method for smoother, more natural results
        radius = 9 if is_dark else 7
        inpainted = cv2.inpaint(roi, dilated, inpaintRadius=radius, flags=cv2.INPAINT_NS)
        
        # Blend the inpainted region with original at mask edges for seamless transition
        blur_k = max(5, min(21, (int(min(roi.shape[:2]) * 0.05) | 1)))
        alpha = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (blur_k, blur_k), 0)
        alpha = np.clip(alpha, 0, 1)
        
        if roi.ndim == 3:
            result = roi.copy()
            for c in range(3):
                result[:,:,c] = (alpha * inpainted[:,:,c] + (1 - alpha) * roi[:,:,c]).astype(np.uint8)
            return result
        else:
            return (alpha * inpainted + (1 - alpha) * roi).astype(np.uint8)
