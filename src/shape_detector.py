import cv2
import numpy as np
from src.shape_classifier import TFClassifier

class ShapeDetector:
    def __init__(self):
        # Initialize our hybrid system
        self.tf_classifier = TFClassifier()
        
    def detect(self, stroke):
        """
        Hybrid detection: 
        1. Try ML model for complex shapes.
        2. Fallback to classical CV for basic geometry.
        """
        if len(stroke) < 10:
            return "UNKNOWN"

        # --- TENSORFLOW PIPELINE ---
        # If we had a model, it would attempt recognition here.
        ml_prediction = self.tf_classifier.predict(stroke)
        if ml_prediction is not None and ml_prediction != 'OTHER':
            return ml_prediction

        # --- CLASSICAL CV FALLBACK PIPELINE ---
        pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
        arc_length = cv2.arcLength(pts, closed=False)
        if arc_length == 0:
            return "UNKNOWN"

        start_pt = np.array(stroke[0])
        end_pt = np.array(stroke[-1])
        start_end_dist = np.linalg.norm(start_pt - end_pt)

        is_closed = start_end_dist < (arc_length * 0.3)

        if not is_closed:
            linearity = arc_length / max(start_end_dist, 1.0)
            if linearity < 1.2:
                return "LINE"
            return "UNKNOWN"

        pts = np.vstack((pts, [pts[0]]))
        closed_perimeter = cv2.arcLength(pts, closed=True)
        area = cv2.contourArea(pts)

        if area < 100:
            return "UNKNOWN"

        epsilon = 0.04 * closed_perimeter
        approx = cv2.approxPolyDP(pts, epsilon, True)
        vertices = len(approx)

        if vertices == 3:
            return "TRIANGLE"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / float(h)
            if 0.75 <= aspect_ratio <= 1.3:
                return "SQUARE"
            else:
                return "RECTANGLE"
        else:
            circularity = (4 * np.pi * area) / (closed_perimeter * closed_perimeter)
            if circularity > 0.7:
                return "CIRCLE"

        return "UNKNOWN"