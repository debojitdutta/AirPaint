import cv2
import sys
from src.camera import Camera
from src.hand_detector import HandDetector

def main():
    print("Initializing Camera...")
    cam = Camera(camera_index=0)
    
    if not cam.is_opened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Initializing MediaPipe Hand Detector...")
    try:
        detector = HandDetector(model_path='models/hand_landmarker.task')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'models/hand_landmarker.task' was downloaded successfully.")
        cam.release()
        sys.exit(1)
        
    print("AirPaint running. Press 'q' to quit.")
    
    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                continue

            # 1. Process the frame through MediaPipe
            detection_result = detector.process(frame)
            
            # 2. Draw the basic hand skeleton
            frame = detector.draw_landmarks(frame, detection_result)
            
            # 3. Find the index fingertip specifically
            h, w, _ = frame.shape
            fingertip = detector.get_index_fingertip(detection_result, w, h)
            
            # 4. Highlight the "brush" (index fingertip) with a bright green circle
            if fingertip:
                cv2.circle(frame, fingertip, 12, (0, 255, 0), -1)
                cv2.circle(frame, fingertip, 12, (255, 255, 255), 2)
            
            cv2.imshow("AirPaint - Hand Tracking Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Securely release hardware and models
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()