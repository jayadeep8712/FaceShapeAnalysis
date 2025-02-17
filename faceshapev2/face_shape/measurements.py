import numpy as np
from .face_shape_enum import FaceShape

class FaceMeasurements:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        
    def calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def get_face_width(self, side_points):
        return self.calculate_distance(
            self.landmarks[side_points[0]],
            self.landmarks[side_points[1]]
        )
    
    def analyze_face_shape(self):
        # Key measurements
        face_length = self.calculate_distance(
            self.landmarks[10],  # Top of forehead
            self.landmarks[152]  # Bottom of chin
        )
        
        forehead_width = self.get_face_width([234, 454])  # Forehead width
        cheekbone_width = self.get_face_width([123, 352])  # Cheekbone width
        jaw_width = self.get_face_width([58, 288])  # Jaw width
        
        # Calculate ratios
        length_width_ratio = face_length / cheekbone_width
        forehead_jaw_ratio = forehead_width / jaw_width
        cheekbone_jaw_ratio = cheekbone_width / jaw_width
        
        # Determine face shape based on ratios
        if length_width_ratio > 1.75:
            return FaceShape.OBLONG
        elif cheekbone_jaw_ratio > 1.3 and forehead_jaw_ratio > 1.3:
            return FaceShape.DIAMOND
        elif forehead_jaw_ratio > 1.3 and cheekbone_jaw_ratio < 1.2:
            return FaceShape.HEART
        elif jaw_width > cheekbone_width * 0.95 and length_width_ratio < 1.3:
            return FaceShape.SQUARE
        elif length_width_ratio < 1.3 and cheekbone_width * 0.95 > jaw_width:
            return FaceShape.ROUND
        else:
            return FaceShape.OVAL