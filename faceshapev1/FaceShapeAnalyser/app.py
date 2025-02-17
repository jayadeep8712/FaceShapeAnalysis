import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List
from flask import Flask, render_template, request, jsonify, Response
import os
from werkzeug.utils import secure_filename

class RealtimeFaceShapeAnalyzer:
    def __init__(self):
        """Initialize the face mesh model for real-time analysis."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # Enable real-time face tracking
            max_num_faces=1,          # Only track one face at a time
            min_detection_confidence=0.5,  # Minimum confidence for detection
            min_tracking_confidence=0.5    # Minimum confidence for tracking
        )

    def calculate_face_measurements(self, landmarks: List[Tuple[float, float]]) -> dict:
        """Calculate key facial measurements."""
        # Key landmark indices for facial features
        jaw_left = landmarks[234]
        jaw_right = landmarks[454]
        chin = landmarks[152]
        forehead = landmarks[10]
        cheek_left = landmarks[123]
        cheek_right = landmarks[352]

        # Calculate facial width, height, and other measurements
        face_width = np.linalg.norm(np.array([jaw_left[0], jaw_left[1]]) - np.array([jaw_right[0], jaw_right[1]]))
        face_height = np.linalg.norm(np.array([chin[0], chin[1]]) - np.array([forehead[0], forehead[1]]))
        cheek_width = np.linalg.norm(np.array([cheek_left[0], cheek_left[1]]) - np.array([cheek_right[0], cheek_right[1]]))
        jaw_width = np.linalg.norm(np.array([landmarks[172][0], landmarks[172][1]]) - np.array([landmarks[397][0], landmarks[397][1]]))

        return {
            'face_width': face_width,
            'face_height': face_height,
            'cheek_width': cheek_width,
            'jaw_width': jaw_width,
            'width_height_ratio': face_width / face_height,
            'jaw_cheek_ratio': jaw_width / cheek_width
        }


    def determine_face_shape(self, measurements: dict) -> Tuple[str, float]:
        """Determine the face shape based on the calculated measurements."""
        width_height_ratio = measurements['width_height_ratio']
        jaw_cheek_ratio = measurements['jaw_cheek_ratio']

        if width_height_ratio > 0.85:
            if jaw_cheek_ratio > 0.9:
                return "Square", 95.0
            else:
                return "Round", 85.0
        else:
            if jaw_cheek_ratio > 0.9:
                return "Rectangle", 90.0
            elif jaw_cheek_ratio < 0.8:
                return "Heart", 80.0
            else:
                if width_height_ratio < 0.75:
                    return "Oval", 75.0
                else:
                    return "Diamond", 70.0   


    def analyze_image(self, image: np.array) -> dict:
        """Analyze the face shape from the given image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) 
                         for landmark in results.multi_face_landmarks[0].landmark]
            
            measurements = self.calculate_face_measurements(landmarks)
            face_shape, confidence = self.determine_face_shape(measurements)
            
            return {
                'face_shape': face_shape,
                'confidence': confidence,
                'measurements': measurements,
                'landmarks': landmarks 
            }
        return None


app = Flask(__name__)

# Initialize the face shape analyzer
analyzer = RealtimeFaceShapeAnalyzer()

# Set up image upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_uploaded_file(file):
    """Ensure that the uploaded file is valid."""
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File not allowed"}), 400
    return None

#Video stream generator for real-time video feed
def generate_video_stream():
    cap = cv2.VideoCapture(0)
    is_streaming = True

    while  is_streaming:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the result from face shape analysis
        result = analyzer.analyze_image(frame)
        
        if result:
            face_shape = result['face_shape']
            confidence = result['confidence']
            measurements = result['measurements']
            landmarks = result['landmarks'] 

            # Add face shape and confidence text to the frame
            cv2.putText(frame, f"Face Shape: {face_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            if face_shape == "Square":
                top_left = (landmarks[234][0], landmarks[10][1])
                bottom_right = (landmarks[454][0], landmarks[152][1])
                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

            elif face_shape == "Rectangle":
                top_left = (landmarks[234][0], landmarks[10][1])
                bottom_right = (landmarks[454][0], landmarks[152][1])
                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)   
                
            elif face_shape == "Diamond":
                top = (landmarks[10][0], landmarks[10][1])
                left = (landmarks[234][0], landmarks[234][1])
                right = (landmarks[454][0], landmarks[454][1])
                bottom = (landmarks[152][0], landmarks[152][1])
                diamond_points = np.array([top, right, bottom, left])
                cv2.polylines(frame, [diamond_points], isClosed=True, color=(225, 255, 0), thickness=2)

            elif face_shape == "Oval":
                center = (landmarks[10][0], (landmarks[10][1] + landmarks[152][1]) // 2)
                axes = (int(measurements['face_width'] // 2), int(measurements['face_height'] // 2))
                cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)

            elif face_shape == "Heart":
                top = (landmarks[10][0], landmarks[10][1])
                left = (landmarks[234][0], landmarks[234][1])
                right = (landmarks[454][0], landmarks[454][1])
                bottom = (landmarks[152][0], landmarks[152][1])
                heart_points = np.array([top, left, bottom, right])
                cv2.polylines(frame, [heart_points], isClosed=True, color=(255, 0, 255), thickness=2)

            elif face_shape == "Round":
                center = ((landmarks[234][0] + landmarks[454][0]) // 2, (landmarks[10][1] + landmarks[152][1]) // 2)
                radius = int(max(measurements['face_width'], measurements['face_height']) // 2)
                cv2.circle(frame, center, radius, (0, 255, 255), 2)

        # Convert frame to JPEG and yield it for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

@app.route('/')
def index():
    """Render the main HTML template."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle the image upload and face shape analysis."""
    file = request.files.get('file')
    validation_error = verify_uploaded_file(file)
    if validation_error:
        return validation_error

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the uploaded image
    image = cv2.imread(filepath)
    result = analyzer.analyze_image(image)
    
    if result:
        face_shape = result['face_shape']
        confidence = result['confidence']
        measurements = result['measurements']
        
        return render_template('index.html', 
                               face_shape=face_shape,
                               confidence=confidence,
                               measurements=measurements,
                               uploaded_image_url=filepath)
    else:
        return jsonify({"error": "No face detected in the image"}), 400

@app.route('/video_feed')
def video_feed():
    """Stream video feed with real-time face shape detection."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
