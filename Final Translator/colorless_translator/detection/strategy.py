"""Page strategy analysis using debate-based classification."""

from typing import Tuple, Dict, Optional, List
import numpy as np


class StrategyAnalyzer:
    """Analyzes manga page type to determine optimal translation strategy."""
    
    STRATEGIES = ["action", "dialogue", "standard"]
    
    def __init__(self, detector):
        self.detector = detector
    
    def analyze_page(
        self, 
        image: np.ndarray
    ) -> Tuple[str, Optional[Dict[str, float]], Dict[str, float]]:
        """
        Analyze page type using strategy debate system.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Tuple of (winning_strategy, adaptive_thresholds, debate_scores)
        """
        results = self.detector.predict_for_analysis(image)
        
        if len(results) == 0 or results[0].boxes is None:
            return "standard", None, {"standard": 1.0, "action": 0.0, "dialogue": 0.0}
        
        stats = self._extract_page_stats(results[0].boxes, image.shape)
        debate_scores = self._run_debate(stats)
        
        winner = max(debate_scores, key=debate_scores.get)
        self._print_debate_results(stats, debate_scores, winner)
        
        return winner, None, debate_scores
    
    def _extract_page_stats(self, boxes, image_shape) -> Dict:
        """Extract statistics from detection results."""
        class_counts = {"bubble": 0, "clean_text": 0, "messy_text": 0, "text_bubble": 0}
        total_area = image_shape[0] * image_shape[1]
        text_area = 0
        
        class_names = ["bubble", "clean_text", "messy_text", "text_bubble"]
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            class_name = class_names[cls_id] if cls_id < len(class_names) else "unknown"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_name in ["clean_text", "messy_text", "text_bubble"]:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                text_area += (x2 - x1) * (y2 - y1)
        
        bubble_count = class_counts["bubble"]
        text_bubble_count = class_counts["text_bubble"]
        clean_text_count = class_counts["clean_text"]
        messy_text_count = class_counts["messy_text"]
        total_text = text_bubble_count + clean_text_count + messy_text_count
        
        if total_text == 0:
            return {
                "bubble_count": bubble_count,
                "text_bubble_count": text_bubble_count,
                "clean_text_count": clean_text_count,
                "messy_text_count": messy_text_count,
                "total_text": 0,
                "text_density": 0,
                "standalone_ratio": 0,
                "messy_ratio": 0,
                "bubble_text_ratio": 0,
            }
        
        return {
            "bubble_count": bubble_count,
            "text_bubble_count": text_bubble_count,
            "clean_text_count": clean_text_count,
            "messy_text_count": messy_text_count,
            "total_text": total_text,
            "text_density": text_area / total_area if total_area > 0 else 0,
            "standalone_ratio": (clean_text_count + messy_text_count) / total_text,
            "messy_ratio": messy_text_count / total_text,
            "bubble_text_ratio": text_bubble_count / bubble_count if bubble_count > 0 else 0,
        }
    
    def _run_debate(self, stats: Dict) -> Dict[str, float]:
        """Run strategy debate and return scores."""
        if stats["total_text"] == 0:
            return {"standard": 1.0, "action": 0.0, "dialogue": 0.0}
        
        scores = {
            "action": self._score_action(stats),
            "dialogue": self._score_dialogue(stats),
            "standard": self._score_standard(stats),
        }
        
        if scores["action"] < 0.4 and scores["dialogue"] < 0.4:
            scores["standard"] += 0.4
        
        return {k: min(v, 1.0) for k, v in scores.items()}
    
    def _score_action(self, stats: Dict) -> float:
        """Calculate action strategy score."""
        score = 0.0
        
        if stats["standalone_ratio"] > 0.4:
            strength = min((stats["standalone_ratio"] - 0.4) / 0.4, 1.0)
            score += 0.4 * strength
        
        if stats["messy_ratio"] > 0.25:
            strength = min((stats["messy_ratio"] - 0.25) / 0.25, 1.0)
            score += 0.3 * strength
        
        if stats["clean_text_count"] > stats["bubble_count"] * 1.5:
            ratio = stats["clean_text_count"] / max(stats["bubble_count"], 1)
            strength = min((ratio - 1.5) / 1.5, 1.0)
            score += 0.3 * strength
        
        return score
    
    def _score_dialogue(self, stats: Dict) -> float:
        """Calculate dialogue strategy score."""
        score = 0.0
        
        if stats["bubble_count"] > 10:
            strength = min((stats["bubble_count"] - 10) / 10, 1.0)
            score += 0.3 * strength
        
        if stats["bubble_text_ratio"] > 0.6:
            strength = (stats["bubble_text_ratio"] - 0.6) / 0.4
            score += 0.4 * strength
        
        if stats["standalone_ratio"] < 0.2:
            strength = (0.2 - stats["standalone_ratio"]) / 0.2
            score += 0.3 * strength
        
        return score
    
    def _score_standard(self, stats: Dict) -> float:
        """Calculate standard strategy score."""
        return 0.3
    
    def _print_debate_results(
        self, 
        stats: Dict, 
        scores: Dict[str, float], 
        winner: str
    ):
        """Print debate results for debugging."""
        print("\n=== STRATEGY DEBATE ===")
        print(f"Page stats: bubbles={stats['bubble_count']}, text_bubble={stats['text_bubble_count']}, "
              f"clean_text={stats['clean_text_count']}, messy_text={stats['messy_text_count']}")
        print(f"Ratios: standalone={stats['standalone_ratio']:.2f}, messy={stats['messy_ratio']:.2f}, "
              f"bubble_text={stats['bubble_text_ratio']:.2f}")
        print("\nStrategy Confidence Scores:")
        
        for strat in self.STRATEGIES:
            score = scores[strat]
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            marker = " ← WINNER" if strat == winner else ""
            print(f"   {strat.upper():12} [{bar}] {score:.2f}{marker}")
