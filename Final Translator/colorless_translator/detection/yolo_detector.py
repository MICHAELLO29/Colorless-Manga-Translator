"""YOLO-based text region detection for manga pages."""

from typing import List, Tuple, Dict, Optional
import numpy as np

from colorless_translator.core.exceptions import DetectionError, ModelLoadError

Detection = Tuple[int, int, int, int, str, float]  # x, y, w, h, class_name, confidence


class YOLODetector:
    """Handles YOLO model loading and text region detection."""
    
    CLASS_NAMES = ["bubble", "clean_text", "messy_text", "text_bubble"]
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_yolov5 = False
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model based on file type."""
        try:
            if "generic" in self.model_path.lower() or self.model_path.endswith("v5.pt"):
                self._load_yolov5()
            else:
                self._load_yolov8()
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model: {e}") from e
    
    def _load_yolov5(self):
        """Load YOLOv5 model via torch hub."""
        import torch
        print("   Detected YOLOv5 model, loading with torch.hub...")
        self.model = torch.hub.load(
            "ultralytics/yolov5", 
            "custom", 
            path=self.model_path, 
            force_reload=False
        )
        self.is_yolov5 = True
        print("YOLOv5 loaded")
    
    def _load_yolov8(self):
        """Load YOLOv8 model via ultralytics."""
        from ultralytics import YOLO
        print("Detected YOLOv8 model, loading with ultralytics...")
        self.model = YOLO(self.model_path)
        self.is_yolov5 = False
        print("YOLOv8 loaded")
    
    def detect(
        self, 
        image: np.ndarray, 
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[Detection]:
        """
        Detect text regions in an image.
        
        Args:
            image: BGR image as numpy array
            thresholds: Per-class confidence thresholds
            
        Returns:
            List of detections as (x, y, w, h, class_name, confidence) tuples
        """
        if thresholds is None:
            thresholds = {name: 0.3 for name in self.CLASS_NAMES}
        
        min_threshold = min(thresholds.values())
        detections = []
        
        try:
            if self.is_yolov5:
                detections = self._detect_yolov5(image, thresholds)
            else:
                detections = self._detect_yolov8(image, min_threshold, thresholds)
        except Exception as e:
            raise DetectionError(f"Detection failed: {e}") from e
        
        return detections
    
    def _detect_yolov5(
        self, 
        image: np.ndarray, 
        thresholds: Dict[str, float]
    ) -> List[Detection]:
        """Run detection with YOLOv5."""
        results = self.model(image, size=640)
        predictions = results.pandas().xyxy[0]
        
        detections = []
        for _, row in predictions.iterrows():
            x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            conf = row["confidence"]
            cls_id = int(row["class"])
            
            class_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else "bubble"
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            min_conf = thresholds.get(class_name, 0.3)
            if conf >= min_conf:
                detections.append((x, y, w, h, class_name, conf))
        
        return detections
    
    def _detect_yolov8(
        self, 
        image: np.ndarray, 
        min_threshold: float,
        thresholds: Dict[str, float]
    ) -> List[Detection]:
        """Run detection with YOLOv8."""
        results = self.model.predict(image, conf=min_threshold, verbose=False, iou=0.5)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                class_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else "unknown"
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                min_conf = thresholds.get(class_name, 0.3)
                if conf >= min_conf:
                    detections.append((x, y, w, h, class_name, conf))
        
        return detections
    
    def predict_for_analysis(
        self, 
        image: np.ndarray, 
        conf: float = 0.15
    ):
        """Run prediction for page type analysis."""
        return self.model.predict(image, conf=conf, verbose=False, iou=0.5)
