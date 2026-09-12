import cv2
import sys
import numpy as np
from src.camera import Camera
from src.hand_detector import HandDetector
from src.canvas import Canvas
from src.drawing_engine import DrawingEngine
from src.shape_detector import ShapeDetector
from src.shape_enhancer import ShapeEnhancer

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
        cam.release()
        sys.exit(1)

    frame = None
    while frame is None:
        frame = cam.read_frame()
        cv2.waitKey(10)

    h, w, _ = frame.shape
    
    canvas = Canvas(width=w, height=h)
    engine = DrawingEngine(canvas=canvas)
    shape_detector = ShapeDetector()
    shape_enhancer = ShapeEnhancer()
    
    last_detected_shape = "None"
        
    print("\n--- AirPaint Controls ---")
    print("1, 2, 3, 4, 5 : Change Colors (Black, Red, Blue, Green, Yellow)")
    print("E             : Eraser Mode")
    print("+ / -         : Change Brush Size")
    print("Q             : Quit")
    
    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                continue

            detection_result = detector.process(frame)
            frame = detector.draw_landmarks(frame, detection_result)
            
            raw_fingertip = detector.get_index_fingertip(detection_result, w, h)
            is_drawing = detector.is_drawing_gesture(detection_result)
            
            smoothed_fingertip, completed_stroke = engine.process_input(is_drawing, raw_fingertip)
            
            if completed_stroke:
                last_detected_shape = shape_detector.detect(completed_stroke)
                enhanced_data = shape_enhancer.enhance(completed_stroke, last_detected_shape)
                engine.apply_enhanced_shape(enhanced_data)
            
            # --- RENDER UI OVERLAYS ---
            if smoothed_fingertip:
                # Render cursor
                cursor_color = engine.get_current_color() if is_drawing else (150, 150, 150)
                cursor_size = engine.get_current_thickness()
                cv2.circle(frame, smoothed_fingertip, cursor_size, cursor_color, -1)
                cv2.circle(frame, smoothed_fingertip, cursor_size, (0,0,0), 1)

            # Display status text
            mode_text = f"Mode: {engine.mode} | Size: {engine.get_current_thickness()}"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Detected: {last_detected_shape}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            combined_view = np.hstack((frame, canvas.image))
            cv2.imshow("AirPaint - Color & Brush System", combined_view)
            
            # --- KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'): engine.set_color((0, 0, 0))       # Black
            elif key == ord('2'): engine.set_color((0, 0, 255))     # Red
            elif key == ord('3'): engine.set_color((255, 0, 0))     # Blue
            elif key == ord('4'): engine.set_color((0, 255, 0))     # Green
            elif key == ord('5'): engine.set_color((0, 255, 255))   # Yellow
            elif key == ord('e'): engine.set_mode("ERASE")          # Eraser
            elif key == ord('=') or key == ord('+'): 
                engine.set_brush_size(engine.brush_size + 2)
            elif key == ord('-'): 
                engine.set_brush_size(engine.brush_size - 2)
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()