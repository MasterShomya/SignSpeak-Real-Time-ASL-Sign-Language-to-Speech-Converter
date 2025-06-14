from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import threading
import time
import os
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob

app = Flask(__name__)

# Global variables
predicted_sentence = ""
current_sign = ""
prediction_count = 0
threshold_frames = 15
last_prediction = ""
camera_active = False

# Create static directory for audio files if it doesn't exist
AUDIO_DIR = os.path.join('static', 'audio')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Load model and encoder (you'll need to place these files in the project directory)
try:
    model = joblib.load("model/asl_model.joblib")
    le = joblib.load("model/label_encoder.joblib")
    model_loaded = True
except FileNotFoundError:
    print("Model files not found. Please ensure asl_model.joblib and label_encoder.joblib are in the project directory.")
    model_loaded = False

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.95,
    min_tracking_confidence=0.95
)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        global predicted_sentence, current_sign, prediction_count, last_prediction
        
        ret, frame = self.video.read()
        if not ret:
            return None
            
        # Flip image for natural feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks and model_loaded:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract 63 features: x, y, z for each landmark
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                # Convert to numpy and reshape for model
                x_input = np.array(features).reshape(1, -1)
                
                # Predict
                y_pred = model.predict(x_input)
                label = le.inverse_transform(y_pred)[0]
                
                # Show current prediction exactly like original code
                current_sign = label
                
                # Stability check over frames
                if label == last_prediction:
                    prediction_count += 1
                else:
                    prediction_count = 0
                    last_prediction = label
                
                # If stable for enough frames, update sentence
                if prediction_count == threshold_frames:
                    if label == "space":
                        predicted_sentence += " "
                    elif label == "del":
                        predicted_sentence = predicted_sentence[:-1]
                    elif label != "nothing":
                        predicted_sentence += label
                    prediction_count = 0  # reset for next sign
        else:
            current_sign = "nothing"
        
        # Add text overlay (like original OpenCV code)
        cv2.rectangle(frame, (10, 10), (630, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {current_sign}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {predicted_sentence}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        return frame

def gen_frames():
    camera = VideoCamera()
    while camera_active:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1/45)  # ~30 FPS

def cleanup_old_audio_files():
    """Remove audio files older than 1 hour"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=1)
        audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        
        for file_path in audio_files:
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Removed old audio file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")

def generate_audio_file(text):
    """Generate MP3 file from text using gTTS"""
    try:
        if not text or text.strip() == "":
            return None
            
        # Clean up old files first
        cleanup_old_audio_files()
        
        # Generate unique filename
        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'Camera stopped'})

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    return jsonify({
        'sentence': predicted_sentence,
        'current_sign': current_sign,
        'prediction_count': prediction_count,
        'threshold_frames': threshold_frames
    })

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global predicted_sentence
    predicted_sentence = ""
    return jsonify({'status': 'Sentence cleared'})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global predicted_sentence
    
    if not predicted_sentence or predicted_sentence.strip() == "":
        return jsonify({
            'status': 'error',
            'message': 'No sentence to speak'
        }), 400
    
    # Generate audio file
    audio_filename = generate_audio_file(predicted_sentence)
    
    if audio_filename:
        return jsonify({
            'status': 'success',
            'sentence': predicted_sentence,
            'audio_url': f'/static/audio/{audio_filename}',
            'message': 'Audio generated successfully'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate audio'
        }), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)