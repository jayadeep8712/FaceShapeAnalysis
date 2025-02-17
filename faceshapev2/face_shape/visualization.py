import cv2
import numpy as np

class FaceVisualizer:
    def __init__(self):
        self.colors = {
            'OVAL': (255, 165, 0),     # Orange
            'ROUND': (0, 255, 0),      # Green
            'SQUARE': (255, 0, 0),     # Blue
            'HEART': (255, 0, 255),    # Magenta
            'DIAMOND': (0, 255, 255),  # Yellow
            'OBLONG': (128, 0, 128)    # Purple
        }
        
    def draw_landmarks(self, image, face_landmarks, face_shape):
        h, w = image.shape[:2]
        landmarks_drawing = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw all landmarks
        for landmark in face_landmarks:
            cv2.circle(landmarks_drawing, 
                      (int(landmark[0] * w), int(landmark[1] * h)),
                      1, self.colors[face_shape.name], -1)
        
        # Draw face shape text
        cv2.putText(image, 
                    f"Face Shape: {face_shape.value}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    self.colors[face_shape.name], 2)
        
        # Blend the landmark drawing with the original image
        return cv2.addWeighted(image, 1, landmarks_drawing, 0.5, 0)