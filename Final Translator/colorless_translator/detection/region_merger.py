"""Region merging and filtering logic for detected text areas."""

from typing import List, Tuple, Set

Detection = Tuple[int, int, int, int, str, float]


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union for two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    
    return inter_area / union_area if union_area != 0 else 0


class RegionMerger:
    """Merges overlapping text regions and associates text with bubbles."""
    
    def __init__(self, iou_threshold: float = 0.1):
        self.iou_threshold = iou_threshold
    
    def merge_overlapping_regions(self, detections: List[Detection]) -> List[Detection]:
        """
        Merge overlapping detections and associate text with bubbles.
        
        Args:
            detections: List of (x, y, w, h, class_name, conf) tuples
            
        Returns:
            Merged list of detections
        """
        bubbles = [d for d in detections if d[4] == "bubble"]
        text_bubbles = [d for d in detections if d[4] == "text_bubble"]
        clean_texts = [d for d in detections if d[4] == "clean_text"]
        messy_texts = [d for d in detections if d[4] == "messy_text"]
        
        merged_regions = []
        used_indices: dict[str, Set[int]] = {
            "text_bubble": set(), 
            "clean_text": set(), 
            "messy_text": set()
        }
        
        for bubble in bubbles:
            bx, by, bw, bh, _, bconf = bubble
            
            contained_text_bubbles = self._find_contained_text_bubbles(
                bubble, text_bubbles, used_indices["text_bubble"]
            )
            
            if contained_text_bubbles:
                merged = self._merge_bubble_with_text(bubble, contained_text_bubbles)
                merged_regions.append(merged)
                for idx, _ in contained_text_bubbles:
                    used_indices["text_bubble"].add(idx)
            else:
                contained_texts = self._find_contained_texts(
                    bubble, clean_texts, messy_texts, used_indices
                )
                
                if contained_texts:
                    merged = self._merge_bubble_with_standalone_text(bubble, contained_texts)
                    merged_regions.append(merged)
                    for text_type, idx, _ in contained_texts:
                        used_indices[text_type].add(idx)
                else:
                    merged_regions.append((bx, by, bw, bh, "bubble_empty", bconf))
        
        merged_regions.extend(
            self._add_standalone_regions(text_bubbles, used_indices["text_bubble"], "standalone_text_bubble", 0.5)
        )
        merged_regions.extend(
            self._add_standalone_regions(clean_texts, used_indices["clean_text"], "standalone_clean_text", 0.0)
        )
        merged_regions.extend(
            self._add_standalone_regions(messy_texts, used_indices["messy_text"], "standalone_messy_text", 0.0)
        )
        
        return merged_regions
    
    def _find_contained_text_bubbles(
        self, 
        bubble: Detection, 
        text_bubbles: List[Detection],
        used: Set[int]
    ) -> List[Tuple[int, Detection]]:
        """Find text_bubble detections contained within a bubble."""
        bx, by, bw, bh, _, _ = bubble
        contained = []
        
        for i, tb in enumerate(text_bubbles):
            if i in used:
                continue
            tx, ty, tw, th, _, _ = tb
            overlap = calculate_iou((bx, by, bw, bh), (tx, ty, tw, th))
            if overlap > 0.3:
                contained.append((i, tb))
        
        return contained
    
    def _find_contained_texts(
        self, 
        bubble: Detection,
        clean_texts: List[Detection],
        messy_texts: List[Detection],
        used_indices: dict
    ) -> List[Tuple[str, int, Detection]]:
        """Find standalone text detections contained within a bubble."""
        bx, by, bw, bh, _, _ = bubble
        contained = []
        
        for text_list, text_type in [(clean_texts, "clean_text"), (messy_texts, "messy_text")]:
            for i, text in enumerate(text_list):
                if i in used_indices[text_type]:
                    continue
                tx, ty, tw, th, _, _ = text
                text_center_x = tx + tw // 2
                text_center_y = ty + th // 2
                
                if bx <= text_center_x <= bx + bw and by <= text_center_y <= by + bh:
                    contained.append((text_type, i, text))
        
        return contained
    
    def _merge_bubble_with_text(
        self, 
        bubble: Detection, 
        text_bubbles: List[Tuple[int, Detection]]
    ) -> Detection:
        """Merge bubble with contained text_bubble detections."""
        bx, by, bw, bh, _, bconf = bubble
        
        best_idx, best_tb = max(text_bubbles, key=lambda x: x[1][5])
        tx, ty, tw, th, _, tconf = best_tb
        
        x_min = min(bx, tx)
        y_min = min(by, ty)
        x_max = max(bx + bw, tx + tw)
        y_max = max(by + bh, ty + th)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min, "bubble_with_text", max(bconf, tconf))
    
    def _merge_bubble_with_standalone_text(
        self, 
        bubble: Detection, 
        texts: List[Tuple[str, int, Detection]]
    ) -> Detection:
        """Merge bubble with standalone text detections."""
        bx, by, bw, bh, _, bconf = bubble
        
        all_boxes = [bubble] + [t[2] for t in texts]
        x_min = min(b[0] for b in all_boxes)
        y_min = min(b[1] for b in all_boxes)
        x_max = max(b[0] + b[2] for b in all_boxes)
        y_max = max(b[1] + b[3] for b in all_boxes)
        max_conf = max(bconf, max(t[2][5] for t in texts))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min, "bubble_with_text", max_conf)
    
    def _add_standalone_regions(
        self,
        detections: List[Detection],
        used: Set[int],
        label: str,
        min_conf: float
    ) -> List[Detection]:
        """Add unused detections as standalone regions."""
        result = []
        for i, det in enumerate(detections):
            if i not in used:
                x, y, w, h, _, conf = det
                if conf > min_conf:
                    result.append((x, y, w, h, label, conf))
        return result
    
    def sort_and_filter(
        self, 
        detections: List[Detection],
        iou_threshold: float = 0.2,
        y_tolerance_ratio: float = 0.5
    ) -> List[Detection]:
        """
        Sort detections in manga reading order and filter duplicates.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for duplicate detection
            y_tolerance_ratio: Tolerance for grouping into rows
            
        Returns:
            Sorted and filtered detections
        """
        detections = sorted(detections, key=lambda d: (d[5], d[2] * d[3]), reverse=True)
        
        filtered = []
        for detection in detections:
            box = detection[:4]
            det_label = detection[4]
            det_group = "bubble" if det_label.startswith("bubble") else ("standalone" if det_label.startswith("standalone") else det_label)
            
            is_duplicate = False
            for fb in filtered:
                fb_box = fb[:4]
                fb_label = fb[4]
                fb_group = "bubble" if fb_label.startswith("bubble") else ("standalone" if fb_label.startswith("standalone") else fb_label)
                
                iou = calculate_iou(box, fb_box)
                
                x, y, w, h = box
                fx, fy, fw, fh = fb_box
                center_x, center_y = x + w / 2, y + h / 2
                fb_center_x, fb_center_y = fx + fw / 2, fy + fh / 2
                center_dist = ((center_x - fb_center_x) ** 2 + (center_y - fb_center_y) ** 2) ** 0.5
                max_dim = max(w, h, fw, fh)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    break

                if det_group == fb_group and center_dist < max_dim * 0.3 and iou > (iou_threshold * 0.35):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return self._sort_reading_order(filtered, y_tolerance_ratio)
    
    def _sort_reading_order(
        self, 
        detections: List[Detection], 
        y_tolerance_ratio: float
    ) -> List[Detection]:
        """Sort detections in manga reading order (right-to-left, top-to-bottom)."""
        rows = []
        remaining = sorted(detections, key=lambda d: d[1])
        
        while remaining:
            base = remaining.pop(0)
            row = [base]
            y_center = base[1] + base[3] / 2
            y_tolerance = base[3] * y_tolerance_ratio
            
            other = []
            for det in remaining:
                det_y_center = det[1] + det[3] / 2
                if abs(y_center - det_y_center) <= y_tolerance:
                    row.append(det)
                else:
                    other.append(det)
            
            remaining = other
            row.sort(key=lambda d: d[0], reverse=True)
            rows.append(row)
        
        return [det for row in rows for det in row]
