import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path='models/hand_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        
            (0, 5), (5, 6), (6, 7), (7, 8),        
            (5, 9), (9, 10), (10, 11), (11, 12),   
            (9, 13), (13, 14), (14, 15), (15, 16), 
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
        ]

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)

    def get_index_fingertip(self, detection_result, frame_width, frame_height):
        if detection_result and detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            index_tip = hand_landmarks[8]
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)
            return (x, y)
        return None
        
    def is_drawing_gesture(self, detection_result):
        """
        Determines if the user intends to draw.
        Added tolerance so the hand doesn't need to be perfectly vertical.
        """
        if not detection_result or not detection_result.hand_landmarks:
            return False
            
        hand_landmarks = detection_result.hand_landmarks[0]
        
        index_tip_y = hand_landmarks[8].y
        index_pip_y = hand_landmarks[6].y
        
        middle_tip_y = hand_landmarks[12].y
        middle_pip_y = hand_landmarks[10].y
        
        # Tolerance: index tip must be significantly higher than the joint
        index_is_up = index_tip_y < (index_pip_y - 0.02)
        
        # Tolerance: middle tip must be clearly folded down
        middle_is_down = middle_tip_y > middle_pip_y
        
        return index_is_up and middle_is_down

    def draw_landmarks(self, frame, detection_result):
        if not detection_result or not detection_result.hand_landmarks:
            return frame

        height, width, _ = frame.shape
        hand_landmarks = detection_result.hand_landmarks[0]

        for connection in self.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (200, 200, 200), -1)

        return frame

    def close(self):
        if self.detector:
            self.detector.close()