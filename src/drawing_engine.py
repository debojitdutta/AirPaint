class DrawingEngine:
    def __init__(self, canvas):
        self.canvas = canvas
        self.brush_size = 6
        self.brush_color = (0, 0, 0) # Default Black
        
        self.mode = "DRAW" # Can be "DRAW" or "ERASE"
        self.eraser_size = 30
        
        self.current_stroke = []
        self.all_strokes = []
        
        self.prev_point = None
        self.smoothed_point = None
        self.smoothing_factor = 0.5
        
        self.frames_since_gesture_lost = 0
        self.debounce_threshold = 5

    def set_color(self, color):
        """Sets the brush color (BGR tuple) and switches to DRAW mode."""
        self.brush_color = color
        self.mode = "DRAW"

    def set_brush_size(self, size):
        self.brush_size = max(2, min(size, 100)) # Clamp between 2 and 100

    def set_mode(self, mode):
        """Switches between 'DRAW' and 'ERASE'."""
        self.mode = mode

    def get_current_thickness(self):
        return self.eraser_size if self.mode == "ERASE" else self.brush_size

    def get_current_color(self):
        return (255, 255, 255) if self.mode == "ERASE" else self.brush_color

    def process_input(self, is_drawing, current_raw_point):
        completed_stroke = None

        if current_raw_point is not None:
            if self.smoothed_point is None:
                self.smoothed_point = current_raw_point
            else:
                new_x = int(self.smoothed_point[0] * (1 - self.smoothing_factor) + current_raw_point[0] * self.smoothing_factor)
                new_y = int(self.smoothed_point[1] * (1 - self.smoothing_factor) + current_raw_point[1] * self.smoothing_factor)
                self.smoothed_point = (new_x, new_y)
        else:
            self.smoothed_point = None

        if is_drawing and self.smoothed_point is not None:
            # Save state only if we are drawing (not erasing) to allow shape enhancement
            if len(self.current_stroke) == 0 and self.mode == "DRAW":
                self.canvas.save_state()

            self.frames_since_gesture_lost = 0
            self.current_stroke.append(self.smoothed_point)
            
            if self.prev_point is not None:
                self.canvas.draw_line(
                    pt1=self.prev_point, 
                    pt2=self.smoothed_point, 
                    color=self.get_current_color(), 
                    thickness=self.get_current_thickness()
                )
                
            self.prev_point = self.smoothed_point

        else:
            self.frames_since_gesture_lost += 1
            
            if self.frames_since_gesture_lost > self.debounce_threshold:
                if len(self.current_stroke) > 0:
                    # We only return completed strokes for enhancement if in DRAW mode
                    if self.mode == "DRAW":
                        completed_stroke = self.current_stroke
                    self.current_stroke = []
                self.prev_point = None

        return self.smoothed_point, completed_stroke

    def apply_enhanced_shape(self, enhanced_shape):
        """Erases the raw stroke and draws the perfectly clean geometric shape."""
        if self.mode != "DRAW":
            return
            
        self.canvas.restore_state()
        shape_type = enhanced_shape.get("type")
        color = self.brush_color
        thickness = self.brush_size
        
        if shape_type == "LINE":
            self.canvas.draw_line(enhanced_shape["pt1"], enhanced_shape["pt2"], color, thickness)
        elif shape_type == "CIRCLE":
            self.canvas.draw_circle(enhanced_shape["center"], enhanced_shape["radius"], color, thickness)
        elif shape_type == "POLYGON":
            self.canvas.draw_polygon(enhanced_shape["points"], color, thickness)
        elif shape_type == "PATH":
            self.canvas.draw_path(enhanced_shape["points"], color, thickness)
            
        self.all_strokes.append(enhanced_shape)