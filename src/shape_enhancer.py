import cv2
import numpy as np

class ShapeEnhancer:
    def enhance(self, stroke, shape_type):
        """
        Takes the raw stroke points and the detected shape type.
        Returns a dictionary containing the instructions to draw the perfect shape.
        """
        pts = np.array(stroke, dtype=np.int32)

        if shape_type == "LINE":
            # Just connect the absolute start and end points
            return {"type": "LINE", "pt1": stroke[0], "pt2": stroke[-1]}

        elif shape_type == "CIRCLE":
            # Find the smallest mathematical circle that completely covers the points
            (x, y), radius = cv2.minEnclosingCircle(pts)
            center = (int(x), int(y))
            radius = int(radius)
            return {"type": "CIRCLE", "center": center, "radius": radius}

        elif shape_type == "RECTANGLE":
            # Find the best-fit rotated bounding box
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            points = [(pt[0], pt[1]) for pt in box]
            return {"type": "POLYGON", "points": points}

        elif shape_type == "SQUARE":
            # Find an upright bounding box, but force width and height to be equal
            x, y, w, h = cv2.boundingRect(pts)
            side = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            half = side // 2
            points = [
                (center_x - half, center_y - half), # Top-Left
                (center_x + half, center_y - half), # Top-Right
                (center_x + half, center_y + half), # Bottom-Right
                (center_x - half, center_y + half)  # Bottom-Left
            ]
            return {"type": "POLYGON", "points": points}

        elif shape_type == "TRIANGLE":
            # Re-approximate the polygon to get exactly 3 clean vertices
            epsilon = 0.04 * cv2.arcLength(pts, True)
            approx = cv2.approxPolyDP(pts, epsilon, True)
            
            if len(approx) >= 3:
                points = [
                    (approx[0][0][0], approx[0][0][1]),
                    (approx[1][0][0], approx[1][0][1]),
                    (approx[2][0][0], approx[2][0][1])
                ]
            else:
                # Fallback if approximation acts weirdly
                points = [stroke[0], stroke[len(stroke)//2], stroke[-1]]
                
            return {"type": "POLYGON", "points": points}

        else:
            # UNKNOWN shape - just keep the messy stroke
            return {"type": "PATH", "points": stroke}