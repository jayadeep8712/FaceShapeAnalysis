import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from face_shape.face_shape_enum import FaceShape
from face_shape.measurements import FaceMeasurements
from face_shape.visualization import FaceVisualizer

class FaceShapeDetector:
    def __init__(self):
        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path= r'E:\master2\models\face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.visualizer = FaceVisualizer()
        
    def process_frame(self, frame):
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            # Get the first face's landmarks
            face_landmarks = detection_result.face_landmarks[0]
            
            # Convert landmarks to numpy array
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
            
            # Analyze face shape
            measurements = FaceMeasurements(landmarks_array)
            face_shape = measurements.analyze_face_shape()
            
            # Visualize results
            frame = self.visualizer.draw_landmarks(frame, landmarks_array, face_shape)
            
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceShapeDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Face Shape Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()