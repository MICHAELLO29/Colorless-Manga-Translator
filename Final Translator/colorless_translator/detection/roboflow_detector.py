"""Roboflow API-based text region detection for manga pages.

Uses the Roboflow Hosted Inference API (via raw REST calls) to detect
speech bubbles and text regions without requiring a local YOLO model file.
"""

import base64
import io
import cv2
import numpy as np
import requests
from typing import List, Tuple, Dict, Optional

from colorless_translator.core.exceptions import DetectionError, ModelLoadError

Detection = Tuple[int, int, int, int, str, float]  # x, y, w, h, class_name, confidence


class RoboflowDetector:
    """Detects text regions using the Roboflow hosted inference API.

    This is a drop-in replacement for YOLODetector that calls the cloud
    API instead of loading a local .pt model file.
    """

    # Map Roboflow class names to internal class names used by the pipeline.
    # The 'bubble-text-detector' model outputs 'bubble' for text-containing
    # bubbles, so we map it to 'text_bubble' which the RegionMerger
    # recognises as a translatable region.
    CLASS_NAME_MAP = {
        "bubble": "text_bubble",
        "clean_text": "clean_text",
        "messy_text": "messy_text",
        "text_bubble": "text_bubble",
        # Common Roboflow label variants
        "text-bubble": "text_bubble",
        "clean-text": "clean_text",
        "messy-text": "messy_text",
        "speech-bubble": "text_bubble",
        "speech_bubble": "text_bubble",
        "text": "clean_text",
    }

    CLASS_NAMES = ["bubble", "clean_text", "messy_text", "text_bubble"]

    API_URL = "https://detect.roboflow.com"

    def __init__(self, api_key: str, model_id: str = "bubble-text-detector-k5qgg/1"):
        """
        Args:
            api_key: Roboflow API key.
            model_id: Roboflow model ID in "project/version" format.
        """
        self.api_key = api_key
        self.model_id = model_id
        # Parse project and version from model_id
        parts = model_id.split("/")
        if len(parts) == 2:
            self.project, self.version = parts
        else:
            self.project = model_id
            self.version = "1"
        self._validate()

    def _validate(self):
        """Validate that the API key looks reasonable."""
        if not self.api_key or self.api_key == "your_roboflow_api_key_here":
            raise ModelLoadError(
                "ROBOFLOW_API_KEY is not set. "
                "Get your API key from https://app.roboflow.com/settings/api-key "
                "and set it in your .env file."
            )
        print(f"   Roboflow detector ready -- model: {self.model_id}")

    # ------------------------------------------------------------------
    # Public API  (mirrors YOLODetector interface)
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Detection]:
        """
        Detect text regions in an image via the Roboflow API.

        Args:
            image: BGR image as numpy array
            thresholds: Per-class confidence thresholds

        Returns:
            List of detections as (x, y, w, h, class_name, confidence) tuples
        """
        if thresholds is None:
            thresholds = {name: 0.20 for name in self.CLASS_NAMES}

        # Hard floor: never send below 0.20 to the API regardless of settings.
        api_conf = max(0.20, min(thresholds.values()))

        try:
            result = self._call_api(image, confidence=api_conf)
        except Exception as e:
            raise DetectionError(f"Roboflow API call failed: {e}") from e

        return self._parse_predictions(result, thresholds)

    def predict_for_analysis(
        self,
        image: np.ndarray,
        conf: float = 0.20,
    ):
        """Run prediction for page type analysis."""
        api_conf = max(0.20, conf)  # never below 0.20
        try:
            result = self._call_api(image, confidence=api_conf)
        except Exception as e:
            raise DetectionError(f"Roboflow analysis call failed: {e}") from e

        return [_RoboflowResultShim(result, self.CLASS_NAME_MAP, self.CLASS_NAMES, api_conf)]

    # ------------------------------------------------------------------
    # REST API call
    # ------------------------------------------------------------------

    def _call_api(self, image: np.ndarray, confidence: float = 0.3) -> dict:
        """Send image to Roboflow hosted inference API and return JSON."""
        # Encode image as JPEG -> base64
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise DetectionError("Failed to encode image for Roboflow API")

        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        url = (
            f"{self.API_URL}/{self.project}/{self.version}"
            f"?api_key={self.api_key}"
            f"&confidence={confidence}"
            f"&overlap=30"
        )

        response = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if response.status_code != 200:
            raise DetectionError(
                f"Roboflow API returned {response.status_code}: {response.text[:200]}"
            )

        return response.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_predictions(
        self,
        result: dict,
        thresholds: Dict[str, float],
    ) -> List[Detection]:
        """Convert Roboflow JSON response to internal Detection tuples."""
        detections: List[Detection] = []
        predictions = result.get("predictions", [])

        for pred in predictions:
            # Roboflow returns center-x, center-y, width, height
            cx = pred.get("x", 0)
            cy = pred.get("y", 0)
            pw = pred.get("width", 0)
            ph = pred.get("height", 0)
            conf = pred.get("confidence", 0.0)
            raw_class = pred.get("class", "bubble")

            # Map to internal class name
            class_name = self.CLASS_NAME_MAP.get(raw_class, raw_class)
            if class_name not in self.CLASS_NAMES:
                # Default unmapped classes to "text_bubble"
                class_name = "text_bubble"

            # Convert center coords -> top-left coords
            x = int(cx - pw / 2)
            y = int(cy - ph / 2)
            w = int(pw)
            h = int(ph)

            min_conf = thresholds.get(class_name, 0.3)
            if conf >= min_conf:
                detections.append((x, y, w, h, class_name, conf))

        return detections


# ======================================================================
# Lightweight shim so StrategyAnalyzer can consume Roboflow results
# without modifications (it accesses result.boxes.cls / .xyxy / etc.)
# ======================================================================


class _FakeBoxes:
    """Mimics ultralytics Boxes object just enough for StrategyAnalyzer."""

    def __init__(self, predictions: list, class_name_map: dict, class_names: list, min_conf: float):
        import torch

        filtered = [p for p in predictions if p.get("confidence", 0) >= min_conf]

        cls_list = []
        xyxy_list = []
        conf_list = []

        for pred in filtered:
            cx = pred.get("x", 0)
            cy = pred.get("y", 0)
            pw = pred.get("width", 0)
            ph = pred.get("height", 0)

            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2

            raw_class = pred.get("class", "bubble")
            mapped = class_name_map.get(raw_class, raw_class)
            if mapped in class_names:
                cls_id = class_names.index(mapped)
            else:
                cls_id = 0  # default to bubble

            cls_list.append(cls_id)
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(pred.get("confidence", 0.0))

        self.cls = torch.tensor(cls_list, dtype=torch.float32) if cls_list else torch.tensor([])
        self.xyxy = torch.tensor(xyxy_list, dtype=torch.float32) if xyxy_list else torch.zeros((0, 4))
        self.conf = torch.tensor(conf_list, dtype=torch.float32) if conf_list else torch.tensor([])

    def __len__(self):
        return len(self.cls)


class _RoboflowResultShim:
    """Mimics a single ultralytics Results object for StrategyAnalyzer."""

    def __init__(self, result: dict, class_name_map: dict, class_names: list, min_conf: float):
        predictions = result.get("predictions", [])
        if predictions:
            self.boxes = _FakeBoxes(predictions, class_name_map, class_names, min_conf)
        else:
            self.boxes = None
