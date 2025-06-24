#!/usr/bin/env python3
"""
Advanced Intelligent Security System - Simplified Face Detection
Real-time monitoring without face-recognition dependency
"""

import sys
import os

# Check dependencies
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    import sqlite3
    import threading
    import time
    import json
    import logging
    import smtplib
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders
    from collections import defaultdict, deque
    import hashlib
    import base64
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("pip install opencv-python numpy Pillow matplotlib")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database management for incidents and analytics"""
    
    def __init__(self, db_path="security_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    type TEXT NOT NULL,
                    confidence REAL,
                    description TEXT,
                    camera_index INTEGER,
                    screenshot_path TEXT,
                    video_path TEXT,
                    zone_id INTEGER,
                    metadata TEXT
                )
            ''')
            
            # System logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    message TEXT,
                    camera_index INTEGER
                )
            ''')
            
            # Detection zones table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    camera_index INTEGER,
                    points TEXT,
                    sensitivity INTEGER DEFAULT 50,
                    enabled BOOLEAN DEFAULT 1,
                    zone_type TEXT DEFAULT 'motion'
                )
            ''')
            
            # Settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def add_incident(self, incident_type, confidence, description, camera_index, 
                    screenshot_path=None, video_path=None, zone_id=None, metadata=None):
        """Add incident to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO incidents (type, confidence, description, camera_index, 
                                     screenshot_path, video_path, zone_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (incident_type, confidence, description, camera_index, 
                  screenshot_path, video_path, zone_id, json.dumps(metadata) if metadata else None))
            
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return incident_id
            
        except Exception as e:
            logger.error(f"Error adding incident: {e}")
            return None
    
    def get_incidents(self, limit=100, camera_index=None, incident_type=None, 
                     start_date=None, end_date=None):
        """Get incidents with filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM incidents WHERE 1=1"
            params = []
            
            if camera_index is not None:
                query += " AND camera_index = ?"
                params.append(camera_index)
            
            if incident_type:
                query += " AND type = ?"
                params.append(incident_type)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            incidents = cursor.fetchall()
            conn.close()
            
            return incidents
            
        except Exception as e:
            logger.error(f"Error getting incidents: {e}")
            return []
    
    def save_setting(self, key, value):
        """Save setting to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES (?, ?)
            ''', (key, json.dumps(value)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving setting: {e}")
    
    def get_setting(self, key, default=None):
        """Get setting from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return default
            
        except Exception as e:
            logger.error(f"Error getting setting: {e}")
            return default

class SimpleFaceDetector:
    """Simple face detection using OpenCV (no face recognition)"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Face tracking
        self.tracked_faces = []
        self.face_id_counter = 0
        self.face_history = deque(maxlen=30)  # Track faces over 30 frames
        
    def detect_faces(self, frame):
        """Detect faces and track them"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detected_faces = []
            
            for (x, y, w, h) in faces:
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Detect eyes within face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                # Detect smile
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.8,
                    minNeighbors=20
                )
                
                # Calculate confidence based on features detected
                confidence = 0.7  # Base confidence
                if len(eyes) >= 2:
                    confidence += 0.2  # Higher confidence if eyes detected
                if len(smiles) > 0:
                    confidence += 0.1  # Slight boost for smile
                
                face_info = {
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'confidence': min(confidence, 1.0),
                    'eyes_detected': len(eyes),
                    'smile_detected': len(smiles) > 0,
                    'area': w * h
                }
                
                detected_faces.append(face_info)
                
                # Draw face rectangle
                color = (0, 255, 0) if len(eyes) >= 2 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                
                # Draw smile
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 255, 0), 2)
                
                # Add labels
                label = f"Person (Eyes: {len(eyes)}, Smile: {len(smiles) > 0})"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add confidence
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Update face history
            self.face_history.append(len(detected_faces))
            
            return frame, detected_faces
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return frame, []
    
    def get_face_statistics(self):
        """Get face detection statistics"""
        if not self.face_history:
            return {'avg_faces': 0, 'max_faces': 0, 'face_frequency': 0}
        
        avg_faces = sum(self.face_history) / len(self.face_history)
        max_faces = max(self.face_history)
        face_frequency = sum(1 for count in self.face_history if count > 0) / len(self.face_history)
        
        return {
            'avg_faces': avg_faces,
            'max_faces': max_faces,
            'face_frequency': face_frequency
        }

class MotionDetectionZones:
    """Advanced motion detection with configurable zones"""
    
    def __init__(self):
        self.zones = {}
        
    def add_zone(self, zone_id, name, points, sensitivity=50, zone_type='motion'):
        """Add detection zone"""
        self.zones[zone_id] = {
            'name': name,
            'points': np.array(points, dtype=np.int32),
            'sensitivity': sensitivity,
            'type': zone_type,
            'enabled': True,
            'last_motion_time': 0,
            'motion_history': deque(maxlen=30)
        }
    
    def detect_motion_in_zones(self, frame, motion_mask):
        """Detect motion within defined zones"""
        motion_events = []
        
        for zone_id, zone in self.zones.items():
            if not zone['enabled']:
                continue
            
            # Create zone mask
            zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(zone_mask, [zone['points']], 255)
            
            # Apply zone mask to motion mask
            zone_motion = cv2.bitwise_and(motion_mask, zone_mask)
            
            # Calculate motion percentage in zone
            total_pixels = cv2.countNonZero(zone_mask)
            motion_pixels = cv2.countNonZero(zone_motion)
            motion_percentage = (motion_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            # Update motion history
            zone['motion_history'].append(motion_percentage)
            
            # Check if motion exceeds threshold
            threshold = zone['sensitivity']
            if motion_percentage > threshold:
                current_time = time.time()
                if current_time - zone['last_motion_time'] > 2:  # Cooldown period
                    motion_events.append({
                        'zone_id': zone_id,
                        'zone_name': zone['name'],
                        'motion_percentage': motion_percentage,
                        'confidence': min(motion_percentage / threshold, 1.0),
                        'zone_type': zone['type']
                    })
                    zone['last_motion_time'] = current_time
            
            # Draw zone on frame
            color = (0, 255, 0) if motion_percentage > threshold else (255, 0, 255)
            cv2.polylines(frame, [zone['points']], True, color, 2)
            cv2.putText(frame, f"{zone['name']}: {motion_percentage:.1f}%", 
                       tuple(zone['points'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, motion_events
    
    def get_zone_statistics(self):
        """Get zone motion statistics"""
        stats = {}
        for zone_id, zone in self.zones.items():
            if zone['motion_history']:
                avg_motion = sum(zone['motion_history']) / len(zone['motion_history'])
                max_motion = max(zone['motion_history'])
                stats[zone_id] = {
                    'name': zone['name'],
                    'avg_motion': avg_motion,
                    'max_motion': max_motion,
                    'sensitivity': zone['sensitivity']
                }
        return stats

class EmailNotificationSystem:
    """Email notification system"""
    
    def __init__(self):
        self.smtp_server = ""
        self.smtp_port = 587
        self.email_user = ""
        self.email_password = ""
        self.recipients = []
        self.enabled = False
    
    def configure(self, smtp_server, smtp_port, email_user, email_password, recipients):
        """Configure email settings"""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password
        self.recipients = recipients if isinstance(recipients, list) else [recipients]
        self.enabled = True
    
    def send_alert(self, subject, body, attachments=None):
        """Send email alert"""
        if not self.enabled:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, self.recipients, text)
            server.quit()
            
            logger.info("Email alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return False

class AdvancedIncidentDetector:
    """Advanced incident detection system"""
    
    def __init__(self):
        # Core components
        self.db_manager = DatabaseManager()
        self.face_detector = SimpleFaceDetector()
        self.motion_zones = MotionDetectionZones()
        self.email_system = EmailNotificationSystem()
        
        # Camera management
        self.cameras = {}
        self.current_camera_index = 0
        self.available_cameras = self.detect_cameras()
        
        # Monitoring state
        self.is_monitoring = False
        self.detection_active = True
        self.recording_active = False
        
        # Detection settings
        self.motion_sensitivity = 50
        self.face_detection_enabled = True
        self.save_clips = True
        self.continuous_recording = False
        
        # Frame processing
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=16, history=500
        )
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Statistics
        self.frame_count = 0
        self.current_fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Recording
        self.video_writers = {}
        self.recording_start_time = None
        
        # Incidents
        self.recent_incidents = deque(maxlen=100)
        self.last_alert_time = {}
        self.alert_cooldown = 30
        
        # Output directories
        self.output_dir = "incidents"
        self.recordings_dir = "recordings"
        self.analytics_dir = "analytics"
        for directory in [self.output_dir, self.recordings_dir, self.analytics_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Advanced Incident Detector initialized")
    
    def detect_cameras(self):
        """Detect available cameras"""
        available_cameras = []
        
        for i in range(8):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.shape[0] > 0:
                        # Get camera info
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        available_cameras.append({
                            'index': i,
                            'resolution': f"{width}x{height}",
                            'fps': fps,
                            'name': f"Camera {i}"
                        })
                        
                        logger.info(f"Found camera {i}: {width}x{height} @ {fps}fps")
                cap.release()
            except Exception as e:
                logger.debug(f"Camera {i} test failed: {e}")
        
        return available_cameras
    
    def start_camera(self, camera_index):
        """Start specific camera"""
        try:
            if camera_index in self.cameras:
                return True
            
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                return False
            
            # Set optimal properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame
            ret, frame = camera.read()
            if ret and frame is not None:
                self.cameras[camera_index] = {
                    'camera': camera,
                    'last_frame': frame,
                    'frame_count': 0,
                    'recording': False
                }
                
                logger.info(f"Camera {camera_index} started successfully")
                return True
            
            camera.release()
            return False
            
        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False
    
    def stop_camera(self, camera_index):
        """Stop specific camera"""
        if camera_index in self.cameras:
            self.cameras[camera_index]['camera'].release()
            if camera_index in self.video_writers:
                self.video_writers[camera_index]['writer'].release()
                del self.video_writers[camera_index]
            del self.cameras[camera_index]
            logger.info(f"Camera {camera_index} stopped")
    
    def start_monitoring(self):
        """Start monitoring system"""
        if not self.available_cameras:
            return False, "No cameras available"
        
        # Start primary camera
        if not self.start_camera(self.current_camera_index):
            return False, f"Failed to start camera {self.current_camera_index}"
        
        self.is_monitoring = True
        
        # Start continuous recording if enabled
        if self.continuous_recording:
            self.start_recording(self.current_camera_index)
        
        logger.info("Monitoring started")
        return True, "Monitoring started successfully"
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        self.is_monitoring = False
        
        # Stop all cameras
        for camera_index in list(self.cameras.keys()):
            self.stop_camera(camera_index)
        
        logger.info("Monitoring stopped")
    
    def start_recording(self, camera_index, duration=None):
        """Start recording for specific camera"""
        if camera_index not in self.cameras:
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.recordings_dir, f"camera_{camera_index}_{timestamp}.avi")
            
            # Get frame dimensions
            frame = self.cameras[camera_index]['last_frame']
            height, width = frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            
            if video_writer.isOpened():
                self.video_writers[camera_index] = {
                    'writer': video_writer,
                    'filename': filename,
                    'start_time': time.time(),
                    'duration': duration
                }
                
                self.cameras[camera_index]['recording'] = True
                logger.info(f"Recording started for camera {camera_index}: {filename}")
                return True
            
        except Exception as e:
            logger.error(f"Recording start error: {e}")
        
        return False
    
    def stop_recording(self, camera_index):
        """Stop recording for specific camera"""
        if camera_index in self.video_writers:
            self.video_writers[camera_index]['writer'].release()
            filename = self.video_writers[camera_index]['filename']
            del self.video_writers[camera_index]
            
            if camera_index in self.cameras:
                self.cameras[camera_index]['recording'] = False
            
            logger.info(f"Recording stopped for camera {camera_index}")
            return filename
        
        return None
    
    def detect_motion(self, frame):
        """Enhanced motion detection"""
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Noise removal
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            total_motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    motion_detected = True
                    total_motion_area += area
                    
                    # Draw motion rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Motion: {int(area)}', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return motion_detected, total_motion_area, fg_mask
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False, 0, np.zeros(frame.shape[:2], dtype=np.uint8)
    
    def process_frame(self, camera_index, frame):
        """Process frame with all detection methods"""
        try:
            if not self.detection_active:
                return frame, []
            
            processed_frame = frame.copy()
            detected_events = []
            
            # Motion detection
            motion_detected, motion_area, motion_mask = self.detect_motion(processed_frame)
            
            # Zone-based motion detection
            if self.motion_zones.zones:
                processed_frame, zone_events = self.motion_zones.detect_motion_in_zones(
                    processed_frame, motion_mask
                )
                detected_events.extend(zone_events)
            
            # Face detection
            if self.face_detection_enabled:
                processed_frame, detected_faces = self.face_detector.detect_faces(processed_frame)
                
                # Process detected faces
                for face in detected_faces:
                    detected_events.append({
                        'type': 'Face Detection',
                        'confidence': face['confidence'],
                        'area': face['area'],
                        'eyes_detected': face['eyes_detected'],
                        'smile_detected': face['smile_detected'],
                        'camera_index': camera_index
                    })
            
            # Motion detection (general)
            if motion_detected and motion_area > (self.motion_sensitivity * 100):
                detected_events.append({
                    'type': 'Motion Detection',
                    'confidence': min(motion_area / 50000, 1.0),
                    'area': motion_area,
                    'camera_index': camera_index
                })
            
            # Add frame overlays
            self.add_frame_overlays(processed_frame, camera_index, detected_events)
            
            # Process detected events
            for event in detected_events:
                self.handle_detection_event(event, frame)
            
            return processed_frame, detected_events
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, []
    
    def add_frame_overlays(self, frame, camera_index, events):
        """Add informational overlays to frame"""
        try:
            # Status overlay
            status_text = "MONITORING" if self.is_monitoring else "STANDBY"
            status_color = (0, 255, 0) if self.is_monitoring else (0, 0, 255)
            cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Camera info
            cv2.putText(frame, f"Camera: {camera_index}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Frame count and FPS
            if camera_index in self.cameras:
                frame_count = self.cameras[camera_index]['frame_count']
                cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Detection status
            detection_status = []
            if self.face_detection_enabled:
                detection_status.append("Face")
            if self.motion_zones.zones:
                detection_status.append("Zones")
            
            if detection_status:
                cv2.putText(frame, f"Detection: {', '.join(detection_status)}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Recording indicator
            if camera_index in self.cameras and self.cameras[camera_index]['recording']:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 55, 38), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Event indicators
            y_offset = 180
            for event in events[-3:]:  # Show last 3 events
                event_text = f"{event['type']}: {event.get('confidence', 0):.2f}"
                cv2.putText(frame, event_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_offset += 25
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Overlay error: {e}")
    
    def handle_detection_event(self, event, frame):
        """Handle detected event"""
        try:
            event_type = event['type']
            camera_index = event['camera_index']
            confidence = event['confidence']
            
            # Check cooldown
            cooldown_key = f"{event_type}_{camera_index}"
            current_time = time.time()
            
            if cooldown_key in self.last_alert_time:
                if current_time - self.last_alert_time[cooldown_key] < self.alert_cooldown:
                    return
            
            self.last_alert_time[cooldown_key] = current_time
            
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.output_dir, f"{event_type}_{camera_index}_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)
            
            # Start incident recording if not already recording
            video_path = None
            if self.save_clips and camera_index in self.cameras and not self.cameras[camera_index]['recording']:
                if self.start_recording(camera_index, duration=30):
                    video_path = self.video_writers[camera_index]['filename']
            
            # Create incident description
            description = f"{event_type} detected on Camera {camera_index}"
            if 'area' in event:
                description += f" - Area: {event['area']}"
            if 'eyes_detected' in event:
                description += f" - Eyes: {event['eyes_detected']}"
            if 'smile_detected' in event:
                description += f" - Smile: {event['smile_detected']}"
            
            # Add to database
            incident_id = self.db_manager.add_incident(
                incident_type=event_type,
                confidence=confidence,
                description=description,
                camera_index=camera_index,
                screenshot_path=screenshot_path,
                video_path=video_path,
                metadata=event
            )
            
            # Add to recent incidents
            incident = {
                'id': incident_id,
                'timestamp': datetime.now(),
                'type': event_type,
                'confidence': confidence,
                'description': description,
                'camera_index': camera_index,
                'screenshot_path': screenshot_path
            }
            
            self.recent_incidents.append(incident)
            
            # Send email alert if configured
            if self.email_system.enabled:
                subject = f"Security Alert: {event_type}"
                body = f"""
Security Alert Detected!

Type: {event_type}
Camera: {camera_index}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Confidence: {confidence:.2f}
Description: {description}

Please check your security system immediately.
                """
                
                self.email_system.send_alert(subject, body, [screenshot_path])
            
            logger.info(f"Event handled: {event_type} - Camera {camera_index} - Confidence: {confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Event handling error: {e}")
    
    def get_frame(self, camera_index):
        """Get processed frame from specific camera"""
        if camera_index not in self.cameras or not self.is_monitoring:
            return None
        
        try:
            camera = self.cameras[camera_index]['camera']
            ret, frame = camera.read()
            
            if ret and frame is not None:
                # Update frame count
                self.cameras[camera_index]['frame_count'] += 1
                self.cameras[camera_index]['last_frame'] = frame
                
                # Calculate FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Process frame
                processed_frame, events = self.process_frame(camera_index, frame)
                
                # Record frame if recording
                if camera_index in self.video_writers:
                    self.video_writers[camera_index]['writer'].write(frame)
                    
                    # Check recording duration
                    writer_info = self.video_writers[camera_index]
                    if writer_info['duration'] and (current_time - writer_info['start_time']) > writer_info['duration']:
                        self.stop_recording(camera_index)
                
                return processed_frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
        
        return None
    
    def get_statistics(self):
        """Get system statistics"""
        stats = {
            'total_incidents': len(self.recent_incidents),
            'incidents_by_type': defaultdict(int),
            'face_stats': self.face_detector.get_face_statistics(),
            'zone_stats': self.motion_zones.get_zone_statistics(),
            'camera_stats': {}
        }
        
        # Count incidents by type
        for incident in self.recent_incidents:
            stats['incidents_by_type'][incident['type']] += 1
        
        # Camera statistics
        for camera_index, camera_info in self.cameras.items():
            stats['camera_stats'][camera_index] = {
                'frame_count': camera_info['frame_count'],
                'recording': camera_info['recording']
            }
        
        return stats
    
    def cleanup(self):
        """Cleanup all resources"""
        self.stop_monitoring()

# Continue with the GUI implementation...
# [The GUI code would continue here similar to the previous version but without face-recognition dependencies]

class SimplifiedSecurityGUI:
    """Simplified GUI without face-recognition dependencies"""
    
    def __init__(self):
        self.detector = AdvancedIncidentDetector()
        self.root = tk.Tk()
        self.setup_gui()
        self.update_thread = None
        self.is_updating = False
        
        # Zone drawing state
        self.drawing_zone = False
        self.zone_points = []
        self.canvas_scale_x = 1.0
        self.canvas_scale_y = 1.0
    
    def setup_gui(self):
        """Setup the GUI"""
        self.root.title("🛡️ Advanced Security System (Simplified)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), background='#1a1a1a', foreground='white')
        style.configure('Status.TLabel', font=('Arial', 10), background='#2d2d2d', foreground='white')
        style.configure('Green.TLabel', font=('Arial', 10, 'bold'), background='#2d2d2d', foreground='#4CAF50')
        style.configure('Red.TLabel', font=('Arial', 10, 'bold'), background='#2d2d2d', foreground='#F44336')
        
        self.create_main_layout()
        self.create_video_panel()
        self.create_control_panel()
        self.create_status_panel()
        self.create_incidents_panel()
        
        # Initialize
        self.update_status_display()
        self.load_settings()
    
    def create_main_layout(self):
        """Create main layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a', height=60)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = ttk.Label(title_frame, text="🛡️ Advanced Security System (No Face Recognition)", style='Title.TLabel')
        title_label.pack(side='left', expand=True, anchor='w')
        
        # Current time
        self.time_label = ttk.Label(title_frame, text="", style='Title.TLabel')
        self.time_label.pack(side='right', anchor='e')
        self.update_time()
        
        # Main content
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel (video + controls)
        self.left_panel = tk.Frame(self.main_frame, bg='#2d2d2d', relief='raised', bd=2)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right panel
        self.right_panel = tk.Frame(self.main_frame, bg='#2d2d2d', width=350, relief='raised', bd=2)
        self.right_panel.pack(side='right', fill='y', padx=(5, 0))
        self.right_panel.pack_propagate(False)
    
    def create_video_panel(self):
        """Create video display panel"""
        video_frame = tk.Frame(self.left_panel, bg='#2d2d2d')
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Camera selection
        camera_select_frame = tk.Frame(video_frame, bg='#2d2d2d')
        camera_select_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(camera_select_frame, text="Camera:", bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold')).pack(side='left')
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_select_frame, textvariable=self.camera_var, state='readonly', width=20)
        self.camera_combo.pack(side='left', padx=(10, 0))
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        # Update camera list
        self.update_camera_list()
        
        # Video display
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=800, height=600)
        self.video_canvas.pack(fill='both', expand=True)
        self.video_canvas.bind('<Button-1>', self.on_canvas_click)
        
        # Video info overlay
        self.video_info_frame = tk.Frame(video_frame, bg='#2d2d2d')
        self.video_info_frame.pack(fill='x', pady=(10, 0))
        
        self.video_info_label = tk.Label(self.video_info_frame, text="Camera Status: Disconnected", 
                                        bg='#2d2d2d', fg='#F44336', font=('Arial', 10))
        self.video_info_label.pack(side='left')
        
        self.recording_indicator = tk.Label(self.video_info_frame, text="", 
                                          bg='#2d2d2d', fg='#F44336', font=('Arial', 10, 'bold'))
        self.recording_indicator.pack(side='right')
    
    def create_control_panel(self):
        """Create control buttons panel"""
        control_frame = tk.Frame(self.left_panel, bg='#2d2d2d', height=100)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Main controls
        main_controls = tk.Frame(control_frame, bg='#2d2d2d')
        main_controls.pack(fill='x', pady=5)
        
        self.start_btn = tk.Button(main_controls, text="🎥 Start Monitoring", 
                                  command=self.start_monitoring, bg='#4CAF50', fg='white', 
                                  font=('Arial', 12, 'bold'), relief='raised', bd=3, width=15)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(main_controls, text="⏹️ Stop Monitoring", 
                                 command=self.stop_monitoring, bg='#F44336', fg='white', 
                                 font=('Arial', 12, 'bold'), relief='raised', bd=3, width=15)
        self.stop_btn.pack(side='left', padx=5)
        
        self.record_btn = tk.Button(main_controls, text="📹 Record", 
                                   command=self.toggle_recording, bg='#FF9800', fg='white', 
                                   font=('Arial', 12, 'bold'), relief='raised', bd=3, width=12)
        self.record_btn.pack(side='left', padx=5)
        
        # Secondary controls
        secondary_controls = tk.Frame(control_frame, bg='#2d2d2d')
        secondary_controls.pack(fill='x', pady=5)
        
        self.detection_btn = tk.Button(secondary_controls, text="🔍 Toggle Detection", 
                                      command=self.toggle_detection, bg='#2196F3', fg='white', 
                                      font=('Arial', 10, 'bold'), relief='raised', bd=2, width=15)
        self.detection_btn.pack(side='left', padx=5)
        
        self.snapshot_btn = tk.Button(secondary_controls, text="📸 Snapshot", 
                                     command=self.take_snapshot, bg='#9C27B0', fg='white', 
                                     font=('Arial', 10, 'bold'), relief='raised', bd=2, width=12)
        self.snapshot_btn.pack(side='left', padx=5)
        
        self.settings_btn = tk.Button(secondary_controls, text="⚙️ Settings", 
                                     command=self.open_settings, bg='#607D8B', fg='white', 
                                     font=('Arial', 10, 'bold'), relief='raised', bd=2, width=12)
        self.settings_btn.pack(side='left', padx=5)
    
    def create_status_panel(self):
        """Create status panel"""
        status_frame = tk.LabelFrame(self.right_panel, text="📊 System Status", 
                                   bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=10)
        
        # Status indicators
        self.monitoring_status = ttk.Label(status_frame, text="Monitoring: OFFLINE", style='Red.TLabel')
        self.monitoring_status.pack(anchor='w', padx=10, pady=5)
        
        self.detection_status = ttk.Label(status_frame, text="Detection: ACTIVE", style='Green.TLabel')
        self.detection_status.pack(anchor='w', padx=10, pady=5)
        
        self.camera_status = ttk.Label(status_frame, text="Camera: Disconnected", style='Red.TLabel')
        self.camera_status.pack(anchor='w', padx=10, pady=5)
        
        self.fps_status = ttk.Label(status_frame, text="FPS: 0.0", style='Status.TLabel')
        self.fps_status.pack(anchor='w', padx=10, pady=5)
        
        self.incidents_status = ttk.Label(status_frame, text="Incidents: 0", style='Status.TLabel')
        self.incidents_status.pack(anchor='w', padx=10, pady=5)
    
    def create_incidents_panel(self):
        """Create incidents panel"""
        incidents_frame = tk.LabelFrame(self.right_panel, text="🚨 Recent Incidents", 
                                      bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        incidents_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Filter controls
        filter_frame = tk.Frame(incidents_frame, bg='#2d2d2d')
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter:", bg='#2d2d2d', fg='white').pack(side='left')
        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, 
                                   values=["All", "Motion Detection", "Face Detection"], 
                                   state='readonly', width=15)
        filter_combo.pack(side='left', padx=(5, 0))
        
        tk.Button(filter_frame, text="🔄", command=self.refresh_incidents, 
                 bg='#2196F3', fg='white', font=('Arial', 8)).pack(side='right', padx=(5, 0))
        
        # Incidents list
        self.incidents_tree = ttk.Treeview(incidents_frame, columns=('Time', 'Type', 'Camera'), 
                                         show='headings', height=15)
        
        self.incidents_tree.heading('Time', text='Time')
        self.incidents_tree.heading('Type', text='Type')
        self.incidents_tree.heading('Camera', text='Camera')
        
        self.incidents_tree.column('Time', width=80)
        self.incidents_tree.column('Type', width=100)
        self.incidents_tree.column('Camera', width=60)
        
        scrollbar = ttk.Scrollbar(incidents_frame, orient="vertical", command=self.incidents_tree.yview)
        self.incidents_tree.configure(yscrollcommand=scrollbar.set)
        
        self.incidents_tree.pack(side='left', fill='both', expand=True, padx=10, pady=5)
        scrollbar.pack(side='right', fill='y')
    
    # Core methods
    def update_time(self):
        """Update current time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_time)
    
    def start_monitoring(self):
        """Start monitoring"""
        try:
            success, message = self.detector.start_monitoring()
            
            if success:
                self.is_updating = True
                self.update_thread = threading.Thread(target=self.update_video_feed, daemon=True)
                self.update_thread.start()
                
                self.update_status_display()
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", f"Failed to start monitoring: {message}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Start monitoring error: {str(e)}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        try:
            self.detector.stop_monitoring()
            self.is_updating = False
            
            if self.update_thread:
                self.update_thread.join(timeout=2)
            
            self.update_status_display()
            
            # Clear video display
            self.video_canvas.delete("all")
            self.video_canvas.create_text(400, 300, text="📹\nCamera Feed\nClick Start to begin", 
                                        fill='white', font=('Arial', 16), justify='center')
            
            messagebox.showinfo("Info", "Monitoring stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Stop monitoring error: {str(e)}")
    
    def update_video_feed(self):
        """Update video feed"""
        while self.is_updating and self.detector.is_monitoring:
            try:
                camera_index = self.get_current_camera_index()
                frame = self.detector.get_frame(camera_index)
                
                if frame is not None:
                    # Convert frame for tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Get canvas dimensions
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        # Resize frame to fit canvas
                        frame_pil = frame_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                        
                        # Convert to PhotoImage
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        # Update GUI in main thread
                        self.root.after(0, self.update_video_display, frame_tk)
                        self.root.after(0, self.update_status_display)
                        self.root.after(0, self.update_incidents_display)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Video update error: {e}")
                time.sleep(0.1)
    
    def update_video_display(self, frame_tk):
        """Update video display"""
        try:
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                self.video_canvas.winfo_width() // 2,
                self.video_canvas.winfo_height() // 2,
                image=frame_tk
            )
            self.video_canvas.image = frame_tk
            
        except Exception as e:
            logger.error(f"Display update error: {e}")
    
    def get_current_camera_index(self):
        """Get current camera index"""
        try:
            camera_str = self.camera_var.get()
            if camera_str and "Camera" in camera_str:
                return int(camera_str.split()[1])
        except:
            pass
        return self.detector.current_camera_index
    
    def update_camera_list(self):
        """Update camera list"""
        camera_names = [f"Camera {cam['index']}" for cam in self.detector.available_cameras]
        self.camera_combo['values'] = camera_names
        
        if camera_names:
            self.camera_combo.set(camera_names[0])
            self.detector.current_camera_index = self.detector.available_cameras[0]['index']
    
    def on_camera_changed(self, event=None):
        """Handle camera change"""
        try:
            camera_index = self.get_current_camera_index()
            self.detector.current_camera_index = camera_index
            self.update_status_display()
        except Exception as e:
            logger.error(f"Camera change error: {e}")
    
    def update_status_display(self):
        """Update status display"""
        try:
            # Monitoring status
            if self.detector.is_monitoring:
                self.monitoring_status.configure(text="Monitoring: ONLINE", style='Green.TLabel')
            else:
                self.monitoring_status.configure(text="Monitoring: OFFLINE", style='Red.TLabel')
            
            # Detection status
            if self.detector.detection_active:
                self.detection_status.configure(text="Detection: ACTIVE", style='Green.TLabel')
            else:
                self.detection_status.configure(text="Detection: DISABLED", style='Red.TLabel')
            
            # Camera status
            camera_index = self.detector.current_camera_index
            if camera_index in self.detector.cameras:
                self.camera_status.configure(text=f"Camera {camera_index}: CONNECTED", style='Green.TLabel')
            else:
                self.camera_status.configure(text=f"Camera {camera_index}: DISCONNECTED", style='Red.TLabel')
            
            # FPS and incidents
            self.fps_status.configure(text=f"FPS: {self.detector.current_fps:.1f}")
            self.incidents_status.configure(text=f"Incidents: {len(self.detector.recent_incidents)}")
            
        except Exception as e:
            logger.error(f"Status update error: {e}")
    
    def update_incidents_display(self):
        """Update incidents display"""
        try:
            # Clear existing items
            for item in self.incidents_tree.get_children():
                self.incidents_tree.delete(item)
            
            # Add recent incidents
            for incident in list(self.detector.recent_incidents)[-20:]:  # Show last 20
                time_str = incident['timestamp'].strftime('%H:%M:%S')
                self.incidents_tree.insert('', 0, values=(
                    time_str,
                    incident['type'],
                    f"Cam {incident['camera_index']}"
                ))
                
        except Exception as e:
            logger.error(f"Incidents update error: {e}")
    
    # Additional methods
    def toggle_detection(self):
        """Toggle detection"""
        self.detector.detection_active = not self.detector.detection_active
        self.update_status_display()
        status = "enabled" if self.detector.detection_active else "disabled"
        messagebox.showinfo("Info", f"Detection {status}")
    
    def toggle_recording(self):
        """Toggle recording"""
        camera_index = self.get_current_camera_index()
        
        if camera_index in self.detector.cameras:
            if self.detector.cameras[camera_index]['recording']:
                self.detector.stop_recording(camera_index)
                messagebox.showinfo("Info", "Recording stopped")
            else:
                if self.detector.start_recording(camera_index):
                    messagebox.showinfo("Info", "Recording started")
                else:
                    messagebox.showerror("Error", "Failed to start recording")
    
    def take_snapshot(self):
        """Take snapshot"""
        camera_index = self.get_current_camera_index()
        
        if camera_index in self.detector.cameras:
            frame = self.detector.cameras[camera_index]['last_frame']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_camera_{camera_index}_{timestamp}.jpg"
            filepath = os.path.join(self.detector.output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            messagebox.showinfo("Success", f"Snapshot saved: {filename}")
    
    def on_canvas_click(self, event):
        """Handle canvas click for zone creation"""
        if self.drawing_zone:
            # Convert canvas coordinates to frame coordinates
            x = int(event.x / self.canvas_scale_x) if self.canvas_scale_x > 0 else event.x
            y = int(event.y / self.canvas_scale_y) if self.canvas_scale_y > 0 else event.y
            
            self.zone_points.append((x, y))
            
            # Draw point on canvas
            self.video_canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                                        fill='red', outline='red')
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#2d2d2d')
        
        # Motion sensitivity
        tk.Label(settings_window, text="Motion Sensitivity:", bg='#2d2d2d', fg='white').pack(pady=5)
        motion_var = tk.IntVar(value=self.detector.motion_sensitivity)
        motion_scale = tk.Scale(settings_window, from_=1, to=100, orient='horizontal', 
                               variable=motion_var, bg='#2d2d2d', fg='white')
        motion_scale.pack(fill='x', padx=20, pady=5)
        
        # Face detection toggle
        face_var = tk.BooleanVar(value=self.detector.face_detection_enabled)
        tk.Checkbutton(settings_window, text="Enable Face Detection", 
                      variable=face_var, bg='#2d2d2d', fg='white').pack(pady=5)
        
        # Save clips toggle
        clips_var = tk.BooleanVar(value=self.detector.save_clips)
        tk.Checkbutton(settings_window, text="Auto-save Incident Clips", 
                      variable=clips_var, bg='#2d2d2d', fg='white').pack(pady=5)
        
        # Save button
        def save_settings():
            self.detector.motion_sensitivity = motion_var.get()
            self.detector.face_detection_enabled = face_var.get()
            self.detector.save_clips = clips_var.get()
            
            # Save to database
            self.detector.db_manager.save_setting('motion_sensitivity', motion_var.get())
            self.detector.db_manager.save_setting('face_detection_enabled', face_var.get())
            self.detector.db_manager.save_setting('save_clips', clips_var.get())
            
            messagebox.showinfo("Success", "Settings saved!")
            settings_window.destroy()
        
        tk.Button(settings_window, text="Save Settings", command=save_settings, 
                 bg='#4CAF50', fg='white', font=('Arial', 12, 'bold')).pack(pady=20)
    
    def load_settings(self):
        """Load settings from database"""
        try:
            self.detector.motion_sensitivity = self.detector.db_manager.get_setting('motion_sensitivity', 50)
            self.detector.face_detection_enabled = self.detector.db_manager.get_setting('face_detection_enabled', True)
            self.detector.save_clips = self.detector.db_manager.get_setting('save_clips', True)
        except Exception as e:
            logger.error(f"Settings load error: {e}")
    
    def refresh_incidents(self):
        """Refresh incidents display"""
        self.update_incidents_display()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("🚀 Starting Simplified Security System GUI...")
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.detector.is_monitoring:
            if messagebox.askokcancel("Quit", "Monitoring is active. Stop monitoring and quit?"):
                self.stop_monitoring()
                self.detector.cleanup()
                self.root.destroy()
        else:
            self.detector.cleanup()
            self.root.destroy()

if __name__ == '__main__':
    try:
        print("🛡️ Starting Advanced Security System (Simplified)")
        print("📹 Initializing cameras...")
        print("🧠 Loading OpenCV face detection...")
        print("📊 Preparing database...")
        print("\n✅ All systems ready!")
        print("🚀 Launching GUI...")
        
        app = SimplifiedSecurityGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
    finally:
        print("✅ Application closed")