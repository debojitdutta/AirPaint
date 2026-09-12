import cv2

class Camera:
    def __init__(self, camera_index=0):
        """
        Initializes the webcam. 
        camera_index=0 is usually the default built-in webcam.
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
    def is_opened(self) -> bool:
        """Check if the camera initialized successfully."""
        return self.cap is not None and self.cap.isOpened()
        
    def read_frame(self):
        """
        Reads a frame from the webcam and mirrors it naturally.
        Returns the processed frame, or None if reading fails.
        """
        if not self.is_opened():
            return None
            
        success, frame = self.cap.read()
        if not success:
            return None
            
        # Mirror the frame horizontally (1 = y-axis)
        mirrored_frame = cv2.flip(frame, 1)
        return mirrored_frame
        
    def release(self):
        """Releases the webcam hardware securely."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None