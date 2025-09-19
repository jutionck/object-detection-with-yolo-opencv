from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import json
import time
import threading
from datetime import datetime
from ultralytics import YOLO
import base64
import numpy as np
from collections import deque
import sys
from frontal_detector import FrontalPersonDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8n.pt")
frontal_detector = FrontalPersonDetector()

class DetectionSystem:
    def __init__(self):
        self.is_running = False
        self.cap = None
        self.detection_data = []
        self.person_counts = deque(maxlen=100)
        self.frontal_person_counts = deque(maxlen=100)
        self.start_time = None
        self.frame_count = 0
        self.total_persons = 0
        self.total_frontal_persons = 0
        self.current_stats = {}
        
    def initialize_video_source(self, source='sample.mp4'):
        if source == 'webcam':
            self.cap = cv2.VideoCapture(0)
        elif source.startswith(('rtsp://', 'http://', 'https://')):
            # IP Camera / NVR support
            self.cap = cv2.VideoCapture(source)
            # Set buffer size to reduce latency for IP cameras
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Set timeout for network streams (5 seconds)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        else:
            # Local video file
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            return False
        return True
    
    def start_detection(self, source='sample.mp4'):
        if self.is_running:
            return False
            
        if not self.initialize_video_source(source):
            return False
            
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.total_persons = 0
        self.total_frontal_persons = 0
        self.detection_data = []
        self.person_counts = deque(maxlen=100)
        self.frontal_person_counts = deque(maxlen=100)
        
        thread = threading.Thread(target=self._detection_loop)
        thread.daemon = True
        thread.start()
        return True
    
    def stop_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        return self._generate_final_report()
    
    def _detection_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.is_running = False
                break
                
            self.frame_count += 1
            current_time = time.time()
            
            frame = cv2.resize(frame, (1280, 720))
            results = model(frame)
            
            # Get frontal person detections
            frontal_persons, all_persons, frontal_results = frontal_detector.filter_frontal_persons(frame, results)
            
            # Create annotated frame with frontal indicators
            annotated_frame = frontal_detector.annotate_frontal_persons(frame, all_persons, frontal_results)
            
            people_count = len(all_persons)
            frontal_people_count = len(frontal_persons)
            
            persons_detected = []
            frontal_persons_detected = []
            
            # Process all persons
            for person in all_persons:
                persons_detected.append({
                    'confidence': person['confidence'],
                    'bbox': person['bbox']
                })
            
            # Process frontal persons
            for person in frontal_persons:
                frontal_persons_detected.append({
                    'confidence': person['confidence'],
                    'bbox': person['bbox'],
                    'face_info': person.get('face_info', {})
                })
            
            self.total_persons += people_count
            self.total_frontal_persons += frontal_people_count
            self.person_counts.append(people_count)
            self.frontal_person_counts.append(frontal_people_count)
            
            detection_record = {
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count,
                'person_count': people_count,
                'frontal_person_count': frontal_people_count,
                'persons': persons_detected,
                'frontal_persons': frontal_persons_detected,
                'elapsed_time': current_time - self.start_time
            }
            self.detection_data.append(detection_record)
            
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            avg_persons = self.total_persons / self.frame_count if self.frame_count > 0 else 0
            avg_frontal_persons = self.total_frontal_persons / self.frame_count if self.frame_count > 0 else 0
            
            self.current_stats = {
                'frame_count': self.frame_count,
                'person_count': people_count,
                'frontal_person_count': frontal_people_count,
                'total_persons': self.total_persons,
                'total_frontal_persons': self.total_frontal_persons,
                'avg_persons': round(avg_persons, 2),
                'avg_frontal_persons': round(avg_frontal_persons, 2),
                'fps': round(fps, 2),
                'elapsed_time': round(elapsed_time, 2),
                'max_persons': max(self.person_counts) if self.person_counts else 0,
                'min_persons': min(self.person_counts) if self.person_counts else 0,
                'max_frontal_persons': max(self.frontal_person_counts) if self.frontal_person_counts else 0,
                'min_frontal_persons': min(self.frontal_person_counts) if self.frontal_person_counts else 0
            }
            
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            socketio.emit('detection_update', {
                'stats': self.current_stats,
                'frame': frame_base64,
                'detection_data': detection_record
            })
            
            time.sleep(0.03)
    
    def _generate_final_report(self):
        if not self.detection_data:
            return None
            
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        report = {
            'summary': {
                'total_frames': self.frame_count,
                'duration_seconds': round(total_duration, 2),
                'average_fps': round(self.frame_count/total_duration, 2),
                'total_persons_detected': self.total_persons,
                'total_frontal_persons_detected': self.total_frontal_persons,
                'average_persons_per_frame': round(self.total_persons/self.frame_count, 2),
                'average_frontal_persons_per_frame': round(self.total_frontal_persons/self.frame_count, 2),
                'max_persons_in_frame': max(self.person_counts) if self.person_counts else 0,
                'min_persons_in_frame': min(self.person_counts) if self.person_counts else 0,
                'max_frontal_persons_in_frame': max(self.frontal_person_counts) if self.frontal_person_counts else 0,
                'min_frontal_persons_in_frame': min(self.frontal_person_counts) if self.frontal_person_counts else 0
            },
            'detailed_data': self.detection_data
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"web_detection_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        return {
            'report': report,
            'filename': filename
        }

detection_system = DetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    data = request.get_json()
    source = data.get('source', 'sample.mp4')
    
    if detection_system.start_detection(source):
        return jsonify({'status': 'success', 'message': 'Detection started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start detection'}), 400

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    result = detection_system.stop_detection()
    if result:
        return jsonify({
            'status': 'success', 
            'message': 'Detection stopped',
            'report': result['report'],
            'filename': result['filename']
        })
    else:
        return jsonify({'status': 'error', 'message': 'No detection session to stop'}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_running': detection_system.is_running,
        'stats': detection_system.current_stats
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=9000)