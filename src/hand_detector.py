import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path='models/hand_landmarker.task'):
        """
        Initializes the MediaPipe Hand Landmarker using the Tasks API.
        We configure it to detect a maximum of 1 hand for optimal performance.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Hardcoding the hand connections explicitly so we don't rely on the 
        # deprecated mp.solutions.hands module
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
        ]

    def process(self, frame):
        """
        Processes an OpenCV BGR frame and returns the detection result.
        """
        # Tasks API expects an RGB image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform hand landmark detection
        return self.detector.detect(mp_image)

    def get_index_fingertip(self, detection_result, frame_width, frame_height):
        """
        Extracts the (x, y) pixel coordinates of the index fingertip (landmark 8).
        Returns a tuple (x, y) or None if no hand is detected.
        """
        if detection_result and detection_result.hand_landmarks:
            # We are only detecting 1 hand, so we take the first item
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Landmark 8 is the index fingertip
            index_tip = hand_landmarks[8]
            
            # Convert normalized coordinates [0.0, 1.0] back to screen pixels
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)
            return (x, y)
            
        return None

    def draw_landmarks(self, frame, detection_result):
        """
        Draws the skeletal hand onto the frame for debugging and visualization.
        """
        if not detection_result or not detection_result.hand_landmarks:
            return frame

        height, width, _ = frame.shape
        hand_landmarks = detection_result.hand_landmarks[0]

        # Draw connecting lines
        for connection in self.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw dots for all 21 landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (200, 200, 200), -1)

        return frame

    def close(self):
        """Releases the MediaPipe detector resources."""
        if self.detector:
            self.detector.close()