#!/usr/bin/env python3
"""
Enterprise Security Intelligence System - Fixed Version
Advanced AI-powered security monitoring with comprehensive features
"""

import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import sqlite3
import threading
import time
import json
import logging
import hashlib
import base64
import zipfile
import shutil
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Fixed email imports
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders
    EMAIL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Email functionality not available: {e}")
    EMAIL_AVAILABLE = False

# Optional advanced imports with fallbacks
try:
    import requests
    WEB_REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: Web requests not available - some notification features disabled")
    WEB_REQUESTS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import seaborn as sns
    import pandas as pd
    ANALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: Advanced analytics not available - install matplotlib, seaborn, pandas")
    ANALYTICS_AVAILABLE = False

try:
    import schedule
    SCHEDULER_AVAILABLE = True
except ImportError:
    print("Warning: Scheduler not available - install schedule package")
    SCHEDULER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDatabaseManager:
    """Enterprise-grade database management with analytics"""
    
    def __init__(self, db_path="enterprise_security.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize comprehensive database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    type TEXT NOT NULL,
                    severity INTEGER DEFAULT 1,
                    confidence REAL,
                    description TEXT,
                    camera_index INTEGER,
                    screenshot_path TEXT,
                    video_path TEXT,
                    zone_id INTEGER,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_by TEXT,
                    resolution_notes TEXT,
                    false_positive BOOLEAN DEFAULT 0,
                    metadata TEXT,
                    hash_signature TEXT UNIQUE
                )
            ''')
            
            # Object detection table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS object_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER,
                    object_type TEXT,
                    confidence REAL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    attributes TEXT,
                    FOREIGN KEY (incident_id) REFERENCES incidents (id)
                )
            ''')
            
            # Security zones table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    camera_index INTEGER,
                    zone_type TEXT,
                    points TEXT,
                    sensitivity INTEGER DEFAULT 50,
                    enabled BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System configuration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configuration (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    description TEXT,
                    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance monitoring
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    camera_fps TEXT,
                    detection_latency REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def add_incident(self, incident_data):
        """Add incident to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create unique hash for incident
            incident_hash = hashlib.md5(
                f"{incident_data['timestamp']}{incident_data['type']}{incident_data['camera_index']}".encode()
            ).hexdigest()
            
            cursor.execute('''
                INSERT INTO incidents (
                    type, severity, confidence, description, camera_index,
                    screenshot_path, video_path, zone_id, metadata, hash_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident_data['type'], incident_data.get('severity', 1),
                incident_data['confidence'], incident_data['description'],
                incident_data['camera_index'], incident_data.get('screenshot_path'),
                incident_data.get('video_path'), incident_data.get('zone_id'),
                json.dumps(incident_data.get('metadata')), incident_hash
            ))
            
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return incident_id
            
        except sqlite3.IntegrityError:
            logger.warning("Duplicate incident detected, skipping")
            return None
        except Exception as e:
            logger.error(f"Error adding incident: {e}")
            return None
    
    def get_incidents(self, limit=100, incident_type=None, start_date=None):
        """Get incidents with filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM incidents WHERE 1=1"
            params = []
            
            if incident_type:
                query += " AND type = ?"
                params.append(incident_type)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            incidents = cursor.fetchall()
            conn.close()
            
            return incidents
            
        except Exception as e:
            logger.error(f"Error getting incidents: {e}")
            return []
    
    def save_setting(self, key, value, category="general"):
        """Save configuration setting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO configuration (key, value, category)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), category))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving setting: {e}")
    
    def get_setting(self, key, default=None):
        """Get configuration setting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT value FROM configuration WHERE key = ?", (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return default
            
        except Exception as e:
            logger.error(f"Error getting setting: {e}")
            return default

class AIObjectDetector:
    """AI object detection with fallback to OpenCV"""
    
    def __init__(self):
        self.detection_method = "opencv"  # fallback method
        self.setup_detection()
        
    def setup_detection(self):
        """Setup object detection"""
        try:
            # Try to load YOLO weights (optional)
            if os.path.exists('models/yolov4.weights') and os.path.exists('models/yolov4.cfg'):
                self.net = cv2.dnn.readNet('models/yolov4.weights', 'models/yolov4.cfg')
                self.output_layers = self.net.getUnconnectedOutLayersNames()
                self.detection_method = "yolo"
                logger.info("YOLO detection loaded")
            else:
                logger.info("YOLO weights not found, using OpenCV detection")
                
        except Exception as e:
            logger.warning(f"Could not load YOLO: {e}")
        
        # Load OpenCV cascades as fallback
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            logger.info("OpenCV cascades loaded")
        except Exception as e:
            logger.error(f"Could not load OpenCV cascades: {e}")
    
    def detect_objects(self, frame):
        """Detect objects using available method"""
        try:
            if self.detection_method == "yolo":
                return self.detect_with_yolo(frame)
            else:
                return self.detect_with_opencv(frame)
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return frame, []
    
    def detect_with_yolo(self, frame):
        """YOLO-based detection"""
        try:
            height, width = frame.shape[:2]
            
            # Prepare input
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            # Process detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            
            detected_objects = []
            class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck']
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else f"object_{class_ids[i]}"
                    confidence = confidences[i]
                    
                    # Draw detection
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detected_objects.append({
                        'type': label,
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
            
            return frame, detected_objects
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return self.detect_with_opencv(frame)
    
    def detect_with_opencv(self, frame):
        """OpenCV-based detection fallback"""
        try:
            detected_objects = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detected_objects.append({
                    'type': 'person',
                    'confidence': 0.8,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
            
            # HOG people detection
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            people, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32))
            for (x, y, w, h) in people:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detected_objects.append({
                    'type': 'person',
                    'confidence': 0.8,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
            
            return frame, detected_objects
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return frame, []

class NotificationSystem:
    """Multi-channel notification system with fallbacks"""
    
    def __init__(self):
        self.email_config = {}
        self.webhook_config = {}
        self.enabled_channels = []
    
    def configure_email(self, smtp_server, smtp_port, username, password, recipients):
        """Configure email notifications"""
        if not EMAIL_AVAILABLE:
            logger.warning("Email not available - skipping email configuration")
            return False
            
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients if isinstance(recipients, list) else [recipients]
        }
        self.enabled_channels.append('email')
        return True
    
    def configure_webhook(self, url, headers=None):
        """Configure webhook notifications"""
        if not WEB_REQUESTS_AVAILABLE:
            logger.warning("Web requests not available - skipping webhook configuration")
            return False
            
        self.webhook_config = {
            'url': url,
            'headers': headers or {}
        }
        self.enabled_channels.append('webhook')
        return True
    
    def send_notification(self, incident_data):
        """Send notification through available channels"""
        results = {}
        
        if 'email' in self.enabled_channels:
            results['email'] = self.send_email_notification(incident_data)
        
        if 'webhook' in self.enabled_channels:
            results['webhook'] = self.send_webhook_notification(incident_data)
        
        # Desktop notification as fallback
        results['desktop'] = self.send_desktop_notification(incident_data)
        
        return results
    
    def send_email_notification(self, incident_data):
        """Send email notification"""
        if not EMAIL_AVAILABLE or not self.email_config:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"🚨 Security Alert: {incident_data['type']}"
            
            # Create email body
            body = f"""
Security Alert Detected!

Type: {incident_data['type']}
Time: {incident_data['timestamp']}
Camera: {incident_data['camera_index']}
Confidence: {incident_data['confidence']:.2f}
Description: {incident_data['description']}

Please check your security system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach screenshot if available
            if incident_data.get('screenshot_path') and os.path.exists(incident_data['screenshot_path']):
                with open(incident_data['screenshot_path'], "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename=alert_screenshot.jpg'
                    )
                    msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['username'], self.email_config['recipients'], text)
            server.quit()
            
            logger.info("Email notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Email notification error: {e}")
            return False
    
    def send_webhook_notification(self, incident_data):
        """Send webhook notification"""
        if not WEB_REQUESTS_AVAILABLE or not self.webhook_config:
            return False
            
        try:
            payload = {
                'event_type': 'security_alert',
                'timestamp': incident_data['timestamp'],
                'incident': incident_data
            }
            
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config['headers'],
                timeout=10
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info("Webhook notification sent successfully")
                return True
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook notification error: {e}")
            return False
    
    def send_desktop_notification(self, incident_data):
        """Send desktop notification as fallback"""
        try:
            # Simple desktop notification using tkinter
            def show_alert():
                alert_window = tk.Toplevel()
                alert_window.title("Security Alert")
                alert_window.geometry("400x200")
                alert_window.configure(bg='red')
                alert_window.attributes('-topmost', True)
                
                tk.Label(alert_window, text="🚨 SECURITY ALERT 🚨", 
                        font=('Arial', 16, 'bold'), bg='red', fg='white').pack(pady=10)
                
                tk.Label(alert_window, text=f"Type: {incident_data['type']}", 
                        font=('Arial', 12), bg='red', fg='white').pack(pady=5)
                
                tk.Label(alert_window, text=f"Camera: {incident_data['camera_index']}", 
                        font=('Arial', 12), bg='red', fg='white').pack(pady=5)
                
                tk.Button(alert_window, text="Acknowledge", 
                         command=alert_window.destroy, font=('Arial', 12)).pack(pady=20)
                
                # Auto-close after 30 seconds
                alert_window.after(30000, alert_window.destroy)
            
            # Schedule the alert to show in main thread
            if hasattr(self, 'root'):
                self.root.after(0, show_alert)
            
            return True
            
        except Exception as e:
            logger.error(f"Desktop notification error: {e}")
            return False

class EnterpriseSecuritySystem:
    """Main enterprise security system"""
    
    def __init__(self):
        # Core components
        self.db_manager = AdvancedDatabaseManager()
        self.ai_detector = AIObjectDetector()
        self.notification_system = NotificationSystem()
        
        # Camera management
        self.cameras = {}
        self.current_camera_index = 0
        self.available_cameras = self.detect_cameras()
        
        # System state
        self.is_monitoring = False
        self.detection_active = True
        
        # Detection settings
        self.motion_sensitivity = 50
        self.ai_detection_enabled = True
        self.face_detection_enabled = True
        
        # Security zones
        self.security_zones = {}
        
        # Performance tracking
        self.frame_count = 0
        self.current_fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Background subtraction for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=16, history=500
        )
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Recording
        self.video_writers = {}
        
        # Incidents
        self.recent_incidents = deque(maxlen=100)
        self.last_alert_time = {}
        self.alert_cooldown = 30
        
        # Setup directories
        for directory in ["incidents", "recordings", "reports", "logs"]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Enterprise Security System initialized")
    
    def detect_cameras(self):
        """Detect available cameras"""
        available_cameras = []
        
        for i in range(8):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.shape[0] > 0:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        available_cameras.append({
                            'index': i,
                            'name': f"Camera {i}",
                            'resolution': f"{width}x{height}",
                            'fps': fps
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
            
            # Set properties
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
        
        try:
            if not self.start_camera(self.current_camera_index):
                return False, f"Failed to start camera {self.current_camera_index}"
            
            self.is_monitoring = True
            
            # Log system start
            self.db_manager.add_incident({
                'type': 'System Start',
                'severity': 1,
                'confidence': 1.0,
                'description': 'Security monitoring started',
                'camera_index': self.current_camera_index,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Monitoring started")
            return True, "Monitoring started successfully"
            
        except Exception as e:
            logger.error(f"Monitoring start error: {e}")
            return False, f"Failed to start monitoring: {str(e)}"
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        try:
            self.is_monitoring = False
            
            # Stop all cameras
            for camera_index in list(self.cameras.keys()):
                self.stop_camera(camera_index)
            
            # Log system stop
            self.db_manager.add_incident({
                'type': 'System Stop',
                'severity': 1,
                'confidence': 1.0,
                'description': 'Security monitoring stopped',
                'camera_index': self.current_camera_index,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Monitoring stopped")
            
        except Exception as e:
            logger.error(f"Monitoring stop error: {e}")
    
    def detect_motion(self, frame):
        """Motion detection"""
        try:
            fg_mask = self.background_subtractor.apply(frame)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            total_motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    motion_detected = True
                    total_motion_area += area
                    
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
            
            # AI object detection
            if self.ai_detection_enabled:
                processed_frame, detected_objects = self.ai_detector.detect_objects(processed_frame)
                
                for obj in detected_objects:
                    detected_events.append({
                        'type': f"{obj['type'].title()} Detection",
                        'confidence': obj['confidence'],
                        'camera_index': camera_index,
                        'object_data': obj
                    })
            
            # Motion events
            if motion_detected and motion_area > (self.motion_sensitivity * 100):
                detected_events.append({
                    'type': 'Motion Detection',
                    'confidence': min(motion_area / 50000, 1.0),
                    'camera_index': camera_index,
                    'motion_area': motion_area
                })
            
            # Add overlays
            self.add_frame_overlays(processed_frame, camera_index, detected_events)
            
            # Handle events
            for event in detected_events:
                self.handle_detection_event(event, frame)
            
            return processed_frame, detected_events
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, []
    
    def add_frame_overlays(self, frame, camera_index, events):
        """Add information overlays to frame"""
        try:
            # Status
            status_text = "MONITORING" if self.is_monitoring else "STANDBY"
            status_color = (0, 255, 0) if self.is_monitoring else (0, 0, 255)
            cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Camera info
            cv2.putText(frame, f"Camera: {camera_index}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Events
            y_offset = 150
            for event in events[-3:]:
                event_text = f"{event['type']}: {event['confidence']:.2f}"
                cv2.putText(frame, event_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_offset += 25
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Overlay error: {e}")
    
    def handle_detection_event(self, event, frame):
        """Handle detection event"""
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
            screenshot_path = os.path.join("incidents", f"{event_type}_{camera_index}_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)
            
            # Create incident
            incident_data = {
                'type': event_type,
                'severity': 2,
                'confidence': confidence,
                'description': f"{event_type} detected on camera {camera_index}",
                'camera_index': camera_index,
                'screenshot_path': screenshot_path,
                'timestamp': datetime.now().isoformat(),
                'metadata': event
            }
            
            # Add to database
            incident_id = self.db_manager.add_incident(incident_data)
            incident_data['id'] = incident_id
            incident_data['timestamp'] = datetime.now()
            
            # Add to recent incidents
            self.recent_incidents.append(incident_data)
            
            # Send notifications
            self.notification_system.send_notification(incident_data)
            
            logger.info(f"Event handled: {event_type} - Camera {camera_index}")
            
        except Exception as e:
            logger.error(f"Event handling error: {e}")
    
    def get_frame(self, camera_index):
        """Get processed frame from camera"""
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
                
                return processed_frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
        
        return None
    
    def cleanup(self):
        """Cleanup system resources"""
        self.stop_monitoring()

class SecuritySystemGUI:
    """Main GUI for the security system"""
    
    def __init__(self):
        self.system = EnterpriseSecuritySystem()
        self.root = tk.Tk()
        self.setup_gui()
        self.update_thread = None
        self.is_updating = False
        
        # Pass root to notification system for desktop alerts
        self.system.notification_system.root = self.root
    
    def setup_gui(self):
        """Setup the GUI"""
        self.root.title("🛡️ Enterprise Security System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
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
        
        title_label = tk.Label(title_frame, text="🛡️ Enterprise Security System", 
                              bg='#1a1a1a', fg='white', font=('Arial', 18, 'bold'))
        title_label.pack(side='left', expand=True, anchor='w')
        
        # Time
        self.time_label = tk.Label(title_frame, text="", bg='#1a1a1a', fg='white', font=('Arial', 14))
        self.time_label.pack(side='right', anchor='e')
        self.update_time()
        
        # Main content
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel
        self.left_panel = tk.Frame(self.main_frame, bg='#2d2d2d', relief='raised', bd=2)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right panel
        self.right_panel = tk.Frame(self.main_frame, bg='#2d2d2d', width=400, relief='raised', bd=2)
        self.right_panel.pack(side='right', fill='y', padx=(5, 0))
        self.right_panel.pack_propagate(False)
    
    def create_video_panel(self):
        """Create video display panel"""
        video_frame = tk.Frame(self.left_panel, bg='#2d2d2d')
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Camera selection
        camera_frame = tk.Frame(video_frame, bg='#2d2d2d')
        camera_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(camera_frame, text="Camera:", bg='#2d2d2d', fg='white', 
                font=('Arial', 12, 'bold')).pack(side='left')
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, 
                                        state='readonly', width=20)
        self.camera_combo.pack(side='left', padx=(10, 0))
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        self.update_camera_list()
        
        # Video display
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=800, height=600)
        self.video_canvas.pack(fill='both', expand=True)
        
        # Video info
        self.video_info_frame = tk.Frame(video_frame, bg='#2d2d2d')
        self.video_info_frame.pack(fill='x', pady=(10, 0))
        
        self.video_info_label = tk.Label(self.video_info_frame, text="Camera Status: Disconnected", 
                                        bg='#2d2d2d', fg='#F44336', font=('Arial', 10))
        self.video_info_label.pack(side='left')
    
    def create_control_panel(self):
        """Create control panel"""
        control_frame = tk.Frame(self.left_panel, bg='#2d2d2d', height=100)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Main controls
        main_controls = tk.Frame(control_frame, bg='#2d2d2d')
        main_controls.pack(fill='x', pady=5)
        
        self.start_btn = tk.Button(main_controls, text="🎥 Start Monitoring", 
                                  command=self.start_monitoring, bg='#4CAF50', fg='white', 
                                  font=('Arial', 12, 'bold'), width=15)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(main_controls, text="⏹️ Stop Monitoring", 
                                 command=self.stop_monitoring, bg='#F44336', fg='white', 
                                 font=('Arial', 12, 'bold'), width=15)
        self.stop_btn.pack(side='left', padx=5)
        
        # Secondary controls
        secondary_controls = tk.Frame(control_frame, bg='#2d2d2d')
        secondary_controls.pack(fill='x', pady=5)
        
        self.detection_btn = tk.Button(secondary_controls, text="🔍 Toggle Detection", 
                                      command=self.toggle_detection, bg='#2196F3', fg='white', 
                                      font=('Arial', 10, 'bold'), width=15)
        self.detection_btn.pack(side='left', padx=5)
        
        self.settings_btn = tk.Button(secondary_controls, text="⚙️ Settings", 
                                     command=self.open_settings, bg='#9C27B0', fg='white', 
                                     font=('Arial', 10, 'bold'), width=12)
        self.settings_btn.pack(side='left', padx=5)
        
        self.reports_btn = tk.Button(secondary_controls, text="📊 Reports", 
                                    command=self.open_reports, bg='#FF9800', fg='white', 
                                    font=('Arial', 10, 'bold'), width=12)
        self.reports_btn.pack(side='left', padx=5)
    
    def create_status_panel(self):
        """Create status panel"""
        status_frame = tk.LabelFrame(self.right_panel, text="📊 System Status", 
                                   bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=10)
        
        self.monitoring_status = tk.Label(status_frame, text="Monitoring: OFFLINE", 
                                         bg='#2d2d2d', fg='#F44336', font=('Arial', 10, 'bold'))
        self.monitoring_status.pack(anchor='w', padx=10, pady=5)
        
        self.detection_status = tk.Label(status_frame, text="Detection: ACTIVE", 
                                        bg='#2d2d2d', fg='#4CAF50', font=('Arial', 10, 'bold'))
        self.detection_status.pack(anchor='w', padx=10, pady=5)
        
        self.camera_status = tk.Label(status_frame, text="Camera: Disconnected", 
                                     bg='#2d2d2d', fg='#F44336', font=('Arial', 10))
        self.camera_status.pack(anchor='w', padx=10, pady=5)
        
        self.fps_status = tk.Label(status_frame, text="FPS: 0.0", 
                                  bg='#2d2d2d', fg='white', font=('Arial', 10))
        self.fps_status.pack(anchor='w', padx=10, pady=5)
        
        self.incidents_status = tk.Label(status_frame, text="Incidents: 0", 
                                        bg='#2d2d2d', fg='white', font=('Arial', 10))
        self.incidents_status.pack(anchor='w', padx=10, pady=5)
    
    def create_incidents_panel(self):
        """Create incidents panel"""
        incidents_frame = tk.LabelFrame(self.right_panel, text="🚨 Recent Incidents", 
                                      bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        incidents_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Incidents list
        self.incidents_tree = ttk.Treeview(incidents_frame, columns=('Time', 'Type', 'Camera'), 
                                         show='headings', height=15)
        
        self.incidents_tree.heading('Time', text='Time')
        self.incidents_tree.heading('Type', text='Type')
        self.incidents_tree.heading('Camera', text='Camera')
        
        self.incidents_tree.column('Time', width=80)
        self.incidents_tree.column('Type', width=120)
        self.incidents_tree.column('Camera', width=60)
        
        scrollbar = ttk.Scrollbar(incidents_frame, orient="vertical", command=self.incidents_tree.yview)
        self.incidents_tree.configure(yscrollcommand=scrollbar.set)
        
        self.incidents_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
    
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_time)
    
    def start_monitoring(self):
        """Start monitoring"""
        try:
            success, message = self.system.start_monitoring()
            
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
            self.system.stop_monitoring()
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
        while self.is_updating and self.system.is_monitoring:
            try:
                camera_index = self.get_current_camera_index()
                frame = self.system.get_frame(camera_index)
                
                if frame is not None:
                    # Convert for tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Get canvas size
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        # Resize frame
                        frame_pil = frame_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        # Update display
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
        return self.system.current_camera_index
    
    def update_camera_list(self):
        """Update camera list"""
        camera_names = [f"Camera {cam['index']}" for cam in self.system.available_cameras]
        self.camera_combo['values'] = camera_names
        
        if camera_names:
            self.camera_combo.set(camera_names[0])
            self.system.current_camera_index = self.system.available_cameras[0]['index']
    
    def on_camera_changed(self, event=None):
        """Handle camera change"""
        try:
            camera_index = self.get_current_camera_index()
            self.system.current_camera_index = camera_index
            self.update_status_display()
        except Exception as e:
            logger.error(f"Camera change error: {e}")
    
    def update_status_display(self):
        """Update status display"""
        try:
            # Monitoring status
            if self.system.is_monitoring:
                self.monitoring_status.configure(text="Monitoring: ONLINE", fg='#4CAF50')
            else:
                self.monitoring_status.configure(text="Monitoring: OFFLINE", fg='#F44336')
            
            # Detection status
            if self.system.detection_active:
                self.detection_status.configure(text="Detection: ACTIVE", fg='#4CAF50')
            else:
                self.detection_status.configure(text="Detection: DISABLED", fg='#F44336')
            
            # Camera status
            camera_index = self.system.current_camera_index
            if camera_index in self.system.cameras:
                self.camera_status.configure(text=f"Camera {camera_index}: CONNECTED", fg='#4CAF50')
            else:
                self.camera_status.configure(text=f"Camera {camera_index}: DISCONNECTED", fg='#F44336')
            
            # FPS and incidents
            self.fps_status.configure(text=f"FPS: {self.system.current_fps:.1f}")
            self.incidents_status.configure(text=f"Incidents: {len(self.system.recent_incidents)}")
            
        except Exception as e:
            logger.error(f"Status update error: {e}")
    
    def update_incidents_display(self):
        """Update incidents display"""
        try:
            # Clear existing
            for item in self.incidents_tree.get_children():
                self.incidents_tree.delete(item)
            
            # Add recent incidents
            for incident in list(self.system.recent_incidents)[-20:]:
                time_str = incident['timestamp'].strftime('%H:%M:%S')
                self.incidents_tree.insert('', 0, values=(
                    time_str,
                    incident['type'],
                    f"Cam {incident['camera_index']}"
                ))
                
        except Exception as e:
            logger.error(f"Incidents update error: {e}")
    
    def toggle_detection(self):
        """Toggle detection"""
        self.system.detection_active = not self.system.detection_active
        self.update_status_display()
        status = "enabled" if self.system.detection_active else "disabled"
        messagebox.showinfo("Info", f"Detection {status}")
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#2d2d2d')
        
        # Motion sensitivity
        tk.Label(settings_window, text="Motion Sensitivity:", bg='#2d2d2d', fg='white').pack(pady=5)
        motion_var = tk.IntVar(value=self.system.motion_sensitivity)
        motion_scale = tk.Scale(settings_window, from_=1, to=100, orient='horizontal', 
                               variable=motion_var, bg='#2d2d2d', fg='white')
        motion_scale.pack(fill='x', padx=20, pady=5)
        
        # AI detection
        ai_var = tk.BooleanVar(value=self.system.ai_detection_enabled)
        tk.Checkbutton(settings_window, text="Enable AI Object Detection", 
                      variable=ai_var, bg='#2d2d2d', fg='white').pack(pady=5)
        
        # Email configuration
        email_frame = tk.LabelFrame(settings_window, text="Email Notifications", 
                                   bg='#2d2d2d', fg='white')
        email_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(email_frame, text="SMTP Server:", bg='#2d2d2d', fg='white').pack(anchor='w')
        smtp_var = tk.StringVar()
        tk.Entry(email_frame, textvariable=smtp_var).pack(fill='x', padx=5, pady=2)
        
        tk.Label(email_frame, text="Email:", bg='#2d2d2d', fg='white').pack(anchor='w')
        email_var = tk.StringVar()
        tk.Entry(email_frame, textvariable=email_var).pack(fill='x', padx=5, pady=2)
        
        # Save function
        def save_settings():
            self.system.motion_sensitivity = motion_var.get()
            self.system.ai_detection_enabled = ai_var.get()
            
            # Configure email if provided
            if smtp_var.get() and email_var.get():
                # Simple email configuration (would need password in real implementation)
                self.system.notification_system.configure_email(
                    smtp_var.get(), 587, email_var.get(), "", [email_var.get()]
                )
            
            # Save to database
            self.system.db_manager.save_setting('motion_sensitivity', motion_var.get())
            self.system.db_manager.save_setting('ai_detection_enabled', ai_var.get())
            
            messagebox.showinfo("Success", "Settings saved!")
            settings_window.destroy()
        
        tk.Button(settings_window, text="Save Settings", command=save_settings, 
                 bg='#4CAF50', fg='white', font=('Arial', 12, 'bold')).pack(pady=20)
    
    def open_reports(self):
        """Open reports window"""
        if not ANALYTICS_AVAILABLE:
            messagebox.showwarning("Reports", "Analytics not available. Install matplotlib and pandas for reporting features.")
            return
            
        reports_window = tk.Toplevel(self.root)
        reports_window.title("Security Reports")
        reports_window.geometry("800x600")
        reports_window.configure(bg='#2d2d2d')
        
        tk.Label(reports_window, text="📊 Security Reports", bg='#2d2d2d', fg='white', 
                font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Report buttons
        tk.Button(reports_window, text="Generate Daily Report", 
                 command=lambda: self.generate_report('daily'), 
                 bg='#4CAF50', fg='white', font=('Arial', 12)).pack(pady=10)
        
        tk.Button(reports_window, text="Generate Weekly Report", 
                 command=lambda: self.generate_report('weekly'), 
                 bg='#2196F3', fg='white', font=('Arial', 12)).pack(pady=10)
        
        tk.Button(reports_window, text="Export Incidents CSV", 
                 command=self.export_incidents_csv, 
                 bg='#FF9800', fg='white', font=('Arial', 12)).pack(pady=10)
    
    def generate_report(self, report_type):
        """Generate security report"""
        try:
            incidents = self.system.db_manager.get_incidents(limit=1000)
            
            if not incidents:
                messagebox.showinfo("Report", "No incidents found for report generation.")
                return
            
            # Simple report generation
            report_text = f"Security Report ({report_type.title()})\n"
            report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_text += f"Total Incidents: {len(incidents)}\n\n"
            
            # Group by type
            incident_types = {}
            for incident in incidents:
                incident_type = incident[2]  # type column
                incident_types[incident_type] = incident_types.get(incident_type, 0) + 1
            
            report_text += "Incidents by Type:\n"
            for incident_type, count in incident_types.items():
                report_text += f"  {incident_type}: {count}\n"
            
            # Save report
            report_path = f"reports/security_report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            messagebox.showinfo("Report", f"Report saved to {report_path}")
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")
    
    def export_incidents_csv(self):
        """Export incidents to CSV"""
        try:
            incidents = self.system.db_manager.get_incidents(limit=1000)
            
            if not incidents:
                messagebox.showinfo("Export", "No incidents found to export.")
                return
            
            # Create CSV content
            csv_content = "ID,Timestamp,Type,Severity,Confidence,Description,Camera,Screenshot\n"
            
            for incident in incidents:
                csv_content += f"{incident[0]},{incident[1]},{incident[2]},{incident[3]},"
                csv_content += f"{incident[4]},{incident[5]},{incident[6]},{incident[7]}\n"
            
            # Save CSV
            csv_path = f"reports/incidents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_path, 'w') as f:
                f.write(csv_content)
            
            messagebox.showinfo("Export", f"Incidents exported to {csv_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export incidents: {str(e)}")
    
    def load_settings(self):
        """Load settings from database"""
        try:
            self.system.motion_sensitivity = self.system.db_manager.get_setting('motion_sensitivity', 50)
            self.system.ai_detection_enabled = self.system.db_manager.get_setting('ai_detection_enabled', True)
        except Exception as e:
            logger.error(f"Settings load error: {e}")
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("🚀 Starting Security System GUI...")
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.system.is_monitoring:
            if messagebox.askokcancel("Quit", "Monitoring is active. Stop monitoring and quit?"):
                self.stop_monitoring()
                self.system.cleanup()
                self.root.destroy()
        else:
            self.system.cleanup()
            self.root.destroy()

if __name__ == '__main__':
    try:
        print("🛡️ Starting Enterprise Security System")
        print("📋 Checking dependencies...")
        
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Email: {'Available' if EMAIL_AVAILABLE else 'Not Available'}")
        print(f"✅ Web Requests: {'Available' if WEB_REQUESTS_AVAILABLE else 'Not Available'}")
        print(f"✅ Analytics: {'Available' if ANALYTICS_AVAILABLE else 'Not Available'}")
        print(f"✅ Scheduler: {'Available' if SCHEDULER_AVAILABLE else 'Not Available'}")
        
        print("\n🚀 Launching Security System...")
        
        app = SecuritySystemGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
    finally:
        print("✅ Application closed")