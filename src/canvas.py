import numpy as np
import cv2

class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.image = self.background.copy()
        
        # State tracking to allow "undoing" the messy stroke
        self.saved_state = self.background.copy()

    def clear(self):
        self.image = self.background.copy()
        self.saved_state = self.background.copy()

    def save_state(self):
        """Saves a snapshot of the canvas before a new stroke begins."""
        self.saved_state = self.image.copy()

    def restore_state(self):
        """Reverts the canvas to the snapshot, erasing the active stroke."""
        self.image = self.saved_state.copy()

    def draw_line(self, pt1, pt2, color, thickness):
        cv2.line(self.image, pt1, pt2, color, thickness, cv2.LINE_AA)

    def draw_circle(self, center, radius, color, thickness):
        cv2.circle(self.image, center, radius, color, thickness, cv2.LINE_AA)

    def draw_polygon(self, points, color, thickness):
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
    def draw_path(self, points, color, thickness):
        """Draws a raw list of connected points (an UNKNOWN shape)."""
        for i in range(1, len(points)):
            self.draw_line(points[i-1], points[i], color, thickness)