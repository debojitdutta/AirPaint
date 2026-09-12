import os
import cv2
import numpy as np

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class TFClassifier:
    def __init__(self, model_path='models/shape_model.keras'):
        """
        Attempts to load a TensorFlow model. 
        If the model does not exist, it operates in 'bypass' mode.
        """
        self.model = None
        self.classes = ['ARROW', 'STAR', 'HEART', 'OTHER'] 
        
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("[ML] TensorFlow model loaded successfully.")
            except Exception as e:
                print(f"[ML] Error loading model: {e}")
        else:
            print("[ML] No TensorFlow model found. Defaulting to classical geometry.")

    def rasterize_stroke(self, stroke, image_size=28):
        """
        Converts a list of (x, y) coordinates into a 28x28 normalized grayscale image.
        This is the standard preprocessing step for CNN image classifiers.
        """
        # Create a blank black image with padding room
        pts = np.array(stroke, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # Create a square bounding box around the stroke
        side = max(w, h) + 20
        canvas = np.zeros((side, side), dtype=np.uint8)
        
        # Shift points so the stroke is centered on our mini canvas
        shift_x = (side - w) // 2 - x
        shift_y = (side - h) // 2 - y
        shifted_pts = pts + [shift_x, shift_y]
        
        # Draw the stroke in white
        for i in range(1, len(shifted_pts)):
            cv2.line(canvas, tuple(shifted_pts[i-1]), tuple(shifted_pts[i]), 255, 2, cv2.LINE_AA)
            
        # Resize to exactly 28x28
        resized = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values between 0.0 and 1.0 and add batch/channel dimensions
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=(0, -1))

    def predict(self, stroke):
        """
        Returns a shape string if confidence is high, else None.
        """
        if self.model is None:
            return None
            
        try:
            # 1. Preprocess
            input_tensor = self.rasterize_stroke(stroke)
            
            # 2. Predict
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            max_index = np.argmax(predictions)
            confidence = predictions[max_index]
            
            # 3. Thresholding
            if confidence > 0.85:
                return self.classes[max_index]
            return None
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return None