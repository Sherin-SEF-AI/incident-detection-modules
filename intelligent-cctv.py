import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
import requests
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist

# Streamlit config
st.set_page_config(
    page_title="Intelligent CCTV Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management"""
    def __init__(self):
        self.storage_dir = Path("storage")
        self.db_path = "events.db"
        self.max_fps = 10
        self.frame_width = 640
        self.frame_height = 480
        self.max_alerts_in_memory = 1000
        
        # Model paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True)
        
        # DNN model files
        self.yolo_weights = self.models_dir / "yolov4.weights"
        self.yolo_config = self.models_dir / "yolov4.cfg"
        self.yolo_classes = self.models_dir / "coco.names"
        
    def get_daily_storage_dir(self, date=None):
        if date is None:
            date = datetime.now()
        daily_dir = self.storage_dir / date.strftime("%Y-%m-%d")
        daily_dir.mkdir(exist_ok=True)
        return daily_dir

    def download_yolo_models(self):
        """Download YOLO models if not present"""
        urls = {
            'yolov4.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
            'yolov4.cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
            'coco.names': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
        }
        
        for filename, url in urls.items():
            filepath = self.models_dir / filename
            if not filepath.exists():
                st.info(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    st.success(f"Downloaded {filename}")
                except Exception as e:
                    st.error(f"Failed to download {filename}: {e}")
                    return False
        return True

class CameraManager:
    """Enhanced camera management with fallbacks"""
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.available_cameras = []
        self.demo_mode = False
        self.demo_frame_count = 0
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.available_cameras = []
        
        # Test camera indices 0-5
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.available_cameras.append(i)
                        st.info(f"✅ Found camera at index {i}")
                cap.release()
            except Exception as e:
                logger.debug(f"Camera {i} test failed: {e}")
        
        return self.available_cameras
    
    def start_camera(self, camera_index=0):
        """Start camera with multiple fallback options"""
        try:
            # Method 1: Try direct index
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                # Method 2: Try different backends
                backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                for backend in backends:
                    try:
                        self.cap = cv2.VideoCapture(camera_index, backend)
                        if self.cap.isOpened():
                            break
                    except Exception as e:
                        logger.debug(f"Backend {backend} failed: {e}")
            
            if not self.cap.isOpened():
                # Method 3: Try other camera indices
                st.warning("🔍 Scanning for available cameras...")
                available = self.detect_cameras()
                if available:
                    self.cap = cv2.VideoCapture(available[0])
                    st.warning(f"Camera {camera_index} not available. Using camera {available[0]} instead.")
            
            if self.cap.isOpened():
                # Test if camera actually works
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 10)
                    self.is_running = True
                    self.demo_mode = False
                    st.success("✅ Camera connected successfully!")
                    return True
                else:
                    self.cap.release()
                    self.cap = None
            
            # Method 4: Enable demo mode with sample video
            st.warning("⚠️ No camera detected. Enabling demo mode with simulated video.")
            self.demo_mode = True
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            st.error(f"Camera error: {e}")
            # Fallback to demo mode
            st.info("🎬 Switching to demo mode...")
            self.demo_mode = True
            self.is_running = True
            return True
    
    def read_frame(self):
        """Read frame with demo mode fallback"""
        if self.demo_mode:
            # Generate demo frame with moving objects
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add background pattern
            frame[:, :] = (20, 30, 40)  # Dark blue background
            
            # Add moving objects for demo
            t = time.time()
            self.demo_frame_count += 1
            
            # Moving person 1
            x1 = int(320 + 200 * np.sin(t * 0.5))
            y1 = int(240 + 100 * np.cos(t * 0.7))
            
            # Moving person 2
            x2 = int(320 + 150 * np.sin(t * 0.3))
            y2 = int(240 + 80 * np.cos(t * 0.4))
            
            # Draw demo "people" rectangles
            cv2.rectangle(frame, (x1-20, y1-40), (x1+20, y1+40), (100, 150, 200), -1)
            cv2.rectangle(frame, (x2-15, y2-35), (x2+15, y2+35), (150, 100, 200), -1)
            
            # Add demo labels
            cv2.putText(frame, f"Person 1", (x1-15, y1-45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Person 2", (x2-15, y2-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add demo info overlay
            cv2.putText(frame, "🎬 DEMO MODE - No Camera Detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {self.demo_frame_count}", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Simulate some noise for realism
            noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            return True, frame
        elif self.cap and self.cap.isOpened():
            return self.cap.read()
        else:
            return False, None
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        self.demo_mode = False
        if self.cap:
            self.cap.release()
            self.cap = None

class PersonTracker:
    """Simple centroid + Kalman filter person tracking"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = 30
        self.max_distance = 50
        
    def create_kalman_filter(self, x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]], dtype=np.float32)
        kf.R *= 10
        kf.Q *= 0.1
        kf.x = np.array([x, y, 0, 0], dtype=np.float32)
        kf.P *= 100
        return kf
        
    def update(self, detections):
        """Update tracks with new detections"""
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return self.tracks
            
        # Extract centroids from detections
        centroids = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centroids.append([cx, cy])
        centroids = np.array(centroids)
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for i, centroid in enumerate(centroids):
                self.tracks[self.next_id] = {
                    'kalman': self.create_kalman_filter(centroid[0], centroid[1]),
                    'disappeared': 0,
                    'position': centroid,
                    'entry_time': time.time()
                }
                self.next_id += 1
        else:
            # Simple distance-based assignment
            track_positions = []
            track_ids = list(self.tracks.keys())
            
            for track_id in track_ids:
                pos = self.tracks[track_id]['kalman'].x[:2]
                track_positions.append(pos)
            
            if len(track_positions) > 0:
                track_positions = np.array(track_positions)
                distances = cdist(track_positions, centroids)
                
                # Greedy assignment
                used_detection_indices = set()
                used_track_indices = set()
                
                # Sort all distance pairs
                distance_pairs = []
                for i in range(len(track_ids)):
                    for j in range(len(centroids)):
                        distance_pairs.append((distances[i, j], i, j))
                distance_pairs.sort()
                
                # Assign greedily
                for dist, track_idx, detection_idx in distance_pairs:
                    if (track_idx not in used_track_indices and 
                        detection_idx not in used_detection_indices and 
                        dist < self.max_distance):
                        
                        track_id = track_ids[track_idx]
                        self.tracks[track_id]['kalman'].predict()
                        self.tracks[track_id]['kalman'].update(centroids[detection_idx])
                        self.tracks[track_id]['position'] = centroids[detection_idx]
                        self.tracks[track_id]['disappeared'] = 0
                        used_track_indices.add(track_idx)
                        used_detection_indices.add(detection_idx)
                
                # Mark unmatched tracks as disappeared
                for i, track_id in enumerate(track_ids):
                    if i not in used_track_indices:
                        self.tracks[track_id]['disappeared'] += 1
                        if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                            del self.tracks[track_id]
                
                # Create new tracks for unmatched detections
                for j in range(len(centroids)):
                    if j not in used_detection_indices:
                        self.tracks[self.next_id] = {
                            'kalman': self.create_kalman_filter(centroids[j][0], centroids[j][1]),
                            'disappeared': 0,
                            'position': centroids[j],
                            'entry_time': time.time()
                        }
                        self.next_id += 1
        
        return self.tracks

class AnomalyDetector:
    """Multi-rule anomaly detection engine"""
    def __init__(self):
        self.people_count_history = []
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.roi_polygon = None
        
    def set_roi(self, points):
        """Set Region of Interest as polygon"""
        if len(points) >= 3:
            self.roi_polygon = np.array(points, dtype=np.int32)
    
    def point_in_roi(self, point):
        """Check if point is inside ROI"""
        if self.roi_polygon is None:
            return True
        return cv2.pointPolygonTest(self.roi_polygon, tuple(point), False) >= 0
    
    def update_people_count(self, count):
        """Update people count history"""
        current_time = time.time()
        self.people_count_history.append((current_time, count))
        
        # Keep only last 5 minutes
        cutoff_time = current_time - 300
        self.people_count_history = [(t, c) for t, c in self.people_count_history if t > cutoff_time]
    
    def check_sudden_change(self, threshold=3, window_seconds=30):
        """Rule A: Sudden change in people count"""
        if len(self.people_count_history) < 2:
            return False
            
        current_time = time.time()
        recent_counts = [c for t, c in self.people_count_history if current_time - t <= window_seconds]
        
        if len(recent_counts) < 2:
            return False
            
        max_change = max(recent_counts) - min(recent_counts)
        return max_change > threshold
    
    def check_loitering(self, tracks, threshold_seconds=180):
        """Rule B: Loitering detection"""
        current_time = time.time()
        for track_id, track in tracks.items():
            if self.point_in_roi(track['position']):
                duration = current_time - track['entry_time']
                if duration > threshold_seconds:
                    return True, track_id
        return False, None
    
    def check_autoencoder_anomaly(self, frame_features):
        """Rule C: Autoencoder-based anomaly (simplified with Isolation Forest)"""
        if not self.is_trained:
            return False
            
        try:
            features_scaled = self.scaler.transform([frame_features])
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            return anomaly_score < -0.5
        except:
            return False
    
    def train_autoencoder(self, feature_history):
        """Train the anomaly detector with normal behavior"""
        if len(feature_history) > 50:
            try:
                features_array = np.array(feature_history)
                self.scaler.fit(features_array)
                features_scaled = self.scaler.transform(features_array)
                self.isolation_forest.fit(features_scaled)
                self.is_trained = True
                return True
            except:
                return False
        return False

class AlertManager:
    """Alert management and external integrations"""
    def __init__(self, config):
        self.config = config
        self.alerts = []
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                rule_id TEXT,
                severity REAL,
                description TEXT,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def create_alert(self, rule_id, severity, description, frame):
        """Create and store alert"""
        timestamp = datetime.now()
        
        # Save alert image
        daily_dir = self.config.get_daily_storage_dir()
        image_filename = f"alert_{timestamp.strftime('%H%M%S_%f')}.jpg"
        image_path = daily_dir / image_filename
        cv2.imwrite(str(image_path), frame)
        
        alert = {
            'timestamp': timestamp,
            'rule_id': rule_id,
            'severity': severity,
            'description': description,
            'image_path': str(image_path),
            'frame': frame
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts in memory
        if len(self.alerts) > self.config.max_alerts_in_memory:
            self.alerts.pop(0)
        
        # Save to database
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, rule_id, severity, description, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp.isoformat(), rule_id, severity, description, str(image_path)))
        conn.commit()
        conn.close()
        
        # Trigger external alerts if needed
        if severity >= 0.8 or rule_id == "weapon":
            self.send_external_alerts(alert)
        
        return alert
    
    def send_external_alerts(self, alert):
        """Send external alerts (SMS, webhook)"""
        try:
            self.send_sms_alert(alert)
            self.send_webhook_alert(alert)
        except Exception as e:
            logger.error(f"Failed to send external alert: {e}")
    
    def send_sms_alert(self, alert):
        """Send SMS via Twilio (placeholder implementation)"""
        logger.info(f"SMS Alert: {alert['description']} at {alert['timestamp']}")
    
    def send_webhook_alert(self, alert):
        """Send webhook alert"""
        try:
            payload = {
                'timestamp': alert['timestamp'].isoformat(),
                'rule_id': alert['rule_id'],
                'severity': alert['severity'],
                'description': alert['description']
            }
            response = requests.post('http://localhost:5055/erss', json=payload, timeout=5)
            logger.info(f"Webhook sent: {response.status_code}")
        except Exception as e:
            logger.warning(f"Webhook failed: {e}")

class TFIDFSearchEngine:
    """TF-IDF based search for forensics (replaces sentence transformers)"""
    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.document_vectors = None
        self.alert_metadata = []
        self.documents = []
        
    def add_alert(self, alert):
        """Add alert to search index"""
        try:
            # Create description for indexing
            description = f"{alert['rule_id']} {alert['description']} {alert['timestamp']}"
            self.documents.append(description)
            self.alert_metadata.append(alert)
            
            # Refit vectorizer with all documents
            if len(self.documents) > 0:
                self.document_vectors = self.vectorizer.fit_transform(self.documents)
        except Exception as e:
            logger.error(f"Failed to add alert to search index: {e}")
    
    def search(self, query, k=5):
        """Search alerts by natural language query"""
        try:
            if self.document_vectors is None or len(self.documents) == 0:
                return []
                
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return results with some similarity
                    result = self.alert_metadata[idx].copy()
                    result['similarity_score'] = float(similarities[idx])
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

class MotionDetector:
    """Fallback motion-based detection when YOLO fails"""
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
    def detect(self, frame):
        """Simple motion-based detection"""
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and aspect ratio (person-like)
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for a person
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w
                    
                    # Person-like aspect ratio (taller than wide)
                    if 1.2 < aspect_ratio < 4.0 and w > 30 and h > 60:
                        detections.append([x, y, x + w, y + h, 0.8, 0, 'person'])
            
            return detections
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return []

class OpenCVDetector:
    """OpenCV DNN-based object detection with motion fallback"""
    def __init__(self, config):
        self.config = config
        self.net = None
        self.classes = []
        self.output_layers = []
        self.motion_detector = MotionDetector()
        self.use_motion_fallback = False
        
    def load_model(self):
        """Load OpenCV DNN model with fallback"""
        try:
            # Try to download and load YOLO models
            if self.config.download_yolo_models():
                # Load class names
                with open(self.config.yolo_classes, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                
                # Load network
                self.net = cv2.dnn.readNet(str(self.config.yolo_weights), str(self.config.yolo_config))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                # Get output layer names
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                
                st.success("✅ YOLO models loaded successfully!")
                return True
            else:
                # Fallback to motion detection
                st.warning("⚠️ YOLO models failed to load. Using motion detection fallback.")
                self.use_motion_fallback = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to load OpenCV DNN model: {e}")
            st.warning(f"Model loading failed: {e}. Using motion detection fallback.")
            self.use_motion_fallback = True
            return True
    
    def detect(self, frame):
        """Detect objects in frame with fallback"""
        if self.use_motion_fallback or self.net is None:
            return self.motion_detector.detect(frame)
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            
            # Run detection
            outputs = self.net.forward(self.output_layers)
            
            # Parse detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    label = self.classes[class_id] if class_id < len(self.classes) else 'unknown'
                    
                    detections.append([x, y, x + w, y + h, confidence, class_id, label])
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed, falling back to motion: {e}")
            self.use_motion_fallback = True
            return self.motion_detector.detect(frame)

class IntelligentCCTV:
    """Main CCTV system class"""
    def __init__(self):
        self.config = Config()
        self.detector = OpenCVDetector(self.config)
        self.tracker = PersonTracker()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(self.config)
        self.search_engine = TFIDFSearchEngine(self.config)
        self.camera_manager = CameraManager()
        
        self.current_frame = None
        self.detections = []
        self.tracks = {}
        self.people_count = 0
        self.people_count_history = []
        
        # Feature history for training
        self.feature_history = []
        
    def load_models(self):
        """Load AI models"""
        return self.detector.load_model()
    
    def start_camera(self, camera_index=0):
        """Start camera capture"""
        return self.camera_manager.start_camera(camera_index)
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_manager.stop_camera()
    
    def extract_frame_features(self, frame):
        """Extract simple features from frame for anomaly detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist_features = hist.flatten() / (hist.sum() + 1e-7)
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Motion features
            motion_feature = np.std(gray)
            
            features = np.concatenate([hist_features, [edge_density, motion_feature]])
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros(18)
    
    def process_frame(self, frame, settings):
        """Process single frame"""
        try:
            # Run object detection
            detections = self.detector.detect(frame)
            
            # Filter and draw detections
            people_detections = []
            weapon_detected = False
            
            for det in detections:
                x1, y1, x2, y2, conf, cls, label = det
                
                # Draw detection
                color = (0, 255, 0) if label == 'person' else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", 
                          (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Track people
                if label == 'person':
                    people_detections.append(det)
                
                # Check for weapons
                if label in ['gun', 'rifle', 'pistol', 'weapon', 'knife', 'scissors']:
                    weapon_detected = True
            
            # Update tracking
            self.tracks = self.tracker.update(people_detections)
            self.people_count = len([t for t in self.tracks.values() if t['disappeared'] == 0])
            
            # Update people count history
            current_time = time.time()
            self.people_count_history.append((current_time, self.people_count))
            if len(self.people_count_history) > 900:
                self.people_count_history.pop(0)
            
            self.anomaly_detector.update_people_count(self.people_count)
            
            # Draw tracks
            for track_id, track in self.tracks.items():
                if track['disappeared'] == 0:
                    pos = track['position'].astype(int)
                    cv2.circle(frame, tuple(pos), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"ID:{track_id}", 
                              (pos[0], pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw ROI if set
            if self.anomaly_detector.roi_polygon is not None:
                cv2.polylines(frame, [self.anomaly_detector.roi_polygon], True, (0, 0, 255), 2)
            
            # Check for anomalies
            alerts_triggered = []
            
            # Rule A: Sudden people count change
            if settings.get('rule_a_enabled', True):
                if self.anomaly_detector.check_sudden_change():
                    alert = self.alert_manager.create_alert(
                        'sudden_change', 0.7, 
                        f'Sudden people count change detected', frame)
                    alerts_triggered.append(alert)
            
            # Rule B: Loitering
            if settings.get('rule_b_enabled', True):
                loitering, track_id = self.anomaly_detector.check_loitering(self.tracks)
                if loitering:
                    alert = self.alert_manager.create_alert(
                        'loitering', 0.6, 
                        f'Loitering detected (Track ID: {track_id})', frame)
                    alerts_triggered.append(alert)
            
            # Rule C: Autoencoder anomaly
            if settings.get('rule_c_enabled', True):
                frame_features = self.extract_frame_features(frame)
                self.feature_history.append(frame_features)
                if len(self.feature_history) > 200:
                    self.feature_history.pop(0)
                
                # Train periodically
                if len(self.feature_history) > 50 and len(self.feature_history) % 50 == 0:
                    self.anomaly_detector.train_autoencoder(self.feature_history)
                
                if self.anomaly_detector.check_autoencoder_anomaly(frame_features):
                    alert = self.alert_manager.create_alert(
                        'autoencoder', 0.8, 
                        'Autoencoder anomaly detected', frame)
                    alerts_triggered.append(alert)
            
            # Weapon detection alert
            if weapon_detected:
                alert = self.alert_manager.create_alert(
                    'weapon', 1.0, 
                    'Weapon detected!', frame)
                alerts_triggered.append(alert)
            
            # Add alerts to search index
            for alert in alerts_triggered:
                self.search_engine.add_alert(alert)
            
            # Display info on frame
            engine_mode = "Motion Detection" if self.detector.use_motion_fallback else "YOLO DNN"
            demo_mode = " (DEMO)" if self.camera_manager.demo_mode else ""
            
            cv2.putText(frame, f"People: {self.people_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Alerts: {len(self.alert_manager.alerts)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Engine: {engine_mode}{demo_mode}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add weapon warning banner
            if weapon_detected:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
                cv2.putText(frame, "⚠️ WEAPON DETECTED!", 
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Check occupancy threshold
            if self.people_count > settings.get('occupancy_limit', 10):
                cv2.rectangle(frame, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), (0, 165, 255), -1)
                cv2.putText(frame, f"OCCUPANCY EXCEEDED: {self.people_count}/{settings.get('occupancy_limit', 10)}", 
                           (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self.current_frame = frame
            self.detections = detections
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return frame

def create_sidebar():
    """Create Streamlit sidebar with controls"""
    st.sidebar.title("🔍 Intelligent CCTV")
    
    settings = {}
    
    # Camera settings
    st.sidebar.subheader("📹 Camera")
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], 0)
    settings['camera_index'] = camera_index
    
    # Detection settings
    st.sidebar.subheader("🎯 Detection")
    settings['occupancy_limit'] = st.sidebar.slider("Occupancy Limit", 1, 50, 10)
    
    # ROI settings
    st.sidebar.subheader("📍 Region of Interest")
    if st.sidebar.button("Set ROI"):
        st.sidebar.info("Click on video to set ROI points, then click 'Finish ROI'")
        settings['setting_roi'] = True
    else:
        settings['setting_roi'] = False
    
    # Rule toggles
    st.sidebar.subheader("⚠️ Anomaly Rules")
    settings['rule_a_enabled'] = st.sidebar.checkbox("Sudden Count Change", True)
    settings['rule_b_enabled'] = st.sidebar.checkbox("Loitering Detection", True)
    settings['rule_c_enabled'] = st.sidebar.checkbox("Autoencoder Anomaly", True)
    
    # Integration settings
    st.sidebar.subheader("📱 Integrations")
    settings['twilio_enabled'] = st.sidebar.checkbox("Twilio SMS", False)
    if settings['twilio_enabled']:
        settings['twilio_sid'] = st.sidebar.text_input("Twilio SID")
        settings['twilio_token'] = st.sidebar.text_input("Twilio Token", type="password")
        settings['twilio_from'] = st.sidebar.text_input("From Number")
        settings['twilio_to'] = st.sidebar.text_input("To Number")
    
    settings['webhook_enabled'] = st.sidebar.checkbox("Webhook Alerts", True)
    
    return settings

def main():
    """Main Streamlit application"""
    st.title("🔍 Intelligent CCTV Dashboard")
    st.info("🖥️ Full-Featured CCTV with Auto-Fallbacks (YOLO → Motion → Demo)")
    
    # Initialize session state
    if 'cctv_system' not in st.session_state:
        st.session_state.cctv_system = IntelligentCCTV()
        st.session_state.system_initialized = False
        st.session_state.camera_started = False
    
    cctv = st.session_state.cctv_system
    settings = create_sidebar()
    
    # Initialize system
    if not st.session_state.system_initialized:
        with st.spinner("🔄 Loading detection models..."):
            if cctv.load_models():
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system.")
                return
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📺 Live Video Feed")
        
        # Camera controls
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("▶️ Start Camera", disabled=st.session_state.camera_started):
                if cctv.start_camera(settings['camera_index']):
                    st.session_state.camera_started = True
                else:
                    st.error("Failed to start camera!")
        
        with col1b:
            if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_started):
                cctv.stop_camera()
                st.session_state.camera_started = False
                st.success("Camera stopped!")
        
        # Video display
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Live video processing
        if st.session_state.camera_started and cctv.camera_manager.is_running:
            try:
                ret, frame = cctv.camera_manager.read_frame()
                if ret and frame is not None:
                    processed_frame = cctv.process_frame(frame, settings)
                    
                    # Convert to RGB for Streamlit (FIXED: use_container_width instead of use_column_width)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    # Status display
                    mode = "Demo Mode" if cctv.camera_manager.demo_mode else "Live Camera"
                    engine = "Motion Detection" if cctv.detector.use_motion_fallback else "YOLO DNN"
                    status_placeholder.success(f"🟢 {mode} Active ({engine})")
                    
                    # Control FPS
                    time.sleep(1.0 / cctv.config.max_fps)
                else:
                    status_placeholder.error("❌ Failed to read frame")
            except Exception as e:
                st.error(f"Video processing error: {e}")
                status_placeholder.error(f"❌ Error: {e}")
        elif st.session_state.camera_started:
            status_placeholder.warning("🟡 Camera starting...")
        else:
            status_placeholder.info("⚪ Camera stopped")
    
    with col2:
        st.subheader("📊 Analytics")
        
        # Live stats
        if st.session_state.camera_started:
            st.metric("👥 Current Occupancy", cctv.people_count)
            st.metric("🚨 Total Alerts", len(cctv.alert_manager.alerts))
            
            # Occupancy chart
            if len(cctv.people_count_history) > 1:
                try:
                    df = pd.DataFrame(cctv.people_count_history, columns=['timestamp', 'count'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df = df.tail(300)
                    
                    st.subheader("📈 15-min Occupancy")
                    st.line_chart(df.set_index('timestamp')['count'])
                except Exception as e:
                    st.warning(f"Chart error: {e}")
        else:
            st.info("Start camera to see live analytics")
        
        # Recent alerts
        st.subheader("🚨 Recent Alerts")
        if len(cctv.alert_manager.alerts) > 0:
            for alert in reversed(cctv.alert_manager.alerts[-5:]):
                with st.expander(f"{alert['rule_id']} - {alert['timestamp'].strftime('%H:%M:%S')}"):
                    st.write(f"**Severity:** {alert['severity']}")
                    st.write(f"**Description:** {alert['description']}")
                    if 'frame' in alert:
                        st.image(cv2.cvtColor(alert['frame'], cv2.COLOR_BGR2RGB))
        else:
            st.info("No alerts yet")
    
    # Forensics search section
    st.subheader("🔍 Forensics Search")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        search_query = st.text_input("Search alerts (natural language):", placeholder="e.g. 'weapon detection last hour'")
        if st.button("🔍 Search") and search_query:
            with st.spinner("Searching..."):
                results = cctv.search_engine.search(search_query, k=10)
                
                if results:
                    st.subheader("🎯 Search Results")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1}: {result['rule_id']} (Score: {result['similarity_score']:.3f})"):
                            st.write(f"**Time:** {result['timestamp']}")
                            st.write(f"**Description:** {result['description']}")
                            if 'frame' in result:
                                st.image(cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB))
                else:
                    st.info("No matching alerts found")
    
    with col4:
        st.subheader("💾 System Status")
        if st.session_state.system_initialized:
            st.success("✅ Detection Engine Ready")
        else:
            st.warning("⏳ Loading models...")
        
        if st.session_state.camera_started:
            mode = "Demo Mode" if cctv.camera_manager.demo_mode else "Live Camera"
            st.success(f"✅ {mode}")
        else:
            st.info("📹 Camera stopped")
        
        # System info
        st.info(f"📁 Storage: {cctv.config.storage_dir}")
        st.info(f"🗄️ Database: {cctv.config.db_path}")
        engine = "Motion Detection" if cctv.detector.use_motion_fallback else "YOLO DNN"
        st.info(f"🤖 Engine: {engine}")
        st.info(f"🔍 Search: TF-IDF")
        
        # Manual alert test
        if st.button("🧪 Test Alert") and cctv.current_frame is not None:
            test_alert = cctv.alert_manager.create_alert(
                'test', 0.5, 'Manual test alert', cctv.current_frame)
            cctv.search_engine.add_alert(test_alert)
            st.success("✅ Test alert created!")
        
        # Clear alerts
        if st.button("🗑️ Clear All Alerts") and len(cctv.alert_manager.alerts) > 0:
            cctv.alert_manager.alerts.clear()
            st.success("✅ Alerts cleared!")

if __name__ == "__main__":
    main()