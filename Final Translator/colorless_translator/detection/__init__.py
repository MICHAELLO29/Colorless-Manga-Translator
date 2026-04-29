"""Detection module for text region identification."""

from colorless_translator.detection.yolo_detector import YOLODetector
from colorless_translator.detection.roboflow_detector import RoboflowDetector
from colorless_translator.detection.region_merger import RegionMerger
from colorless_translator.detection.strategy import StrategyAnalyzer

__all__ = ["YOLODetector", "RoboflowDetector", "RegionMerger", "StrategyAnalyzer"]
