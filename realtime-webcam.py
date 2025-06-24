#!/usr/bin/env python3
"""
Intelligent Incident Detection System - Tkinter GUI
Real-time webcam monitoring with motion detection and alerts
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import os
import json
from datetime import datetime
from PIL import Image, ImageTk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentDetector:
    def __init__(self):
        self.camera = None
        self.is_monitoring = False
        self.detection_active = True
        self.motion_threshold = 5000
        self.last_alert_time = 0
        self.alert_cooldown = 30
        
        # Detection settings
        self.motion_sensitivity = 50
        self.audio_sensitivity = 30
        self.save_clips = True
        self.email_alerts = False
        
        # Storage
        self.incidents = []
        self.output_dir = "incidents"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=16, history=500
        )
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Video recording
        self.video_writer = None
        self.recording = False
        self.recording_start_time = None
        
        # Frame management
        self.current_frame = None
        self.frame_count = 0
        self.fps_count = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Camera detection
        self.available_cameras = self.detect_cameras()
        self.camera_index = self.available_cameras[0] if self.available_cameras else 0
        
        logger.info(f"Initialized with {len(self.available_cameras)} cameras available")

    def detect_cameras(self):
        """Detect available cameras"""
        available_cameras = []
        
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.shape[0] > 0:
                        available_cameras.append(i)
                        logger.info(f"✅ Camera {i} working - Resolution: {frame.shape}")
                cap.release()
            except Exception as e:
                logger.debug(f"Camera {i} test failed: {e}")
        
        return available_cameras
    
    def start_camera(self):
        """Start camera capture"""
        if not self.available_cameras:
            return False
        
        try:
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            
            if not self.camera.isOpened():
                return False
            
            # Set optimal properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self.current_frame = frame
                logger.info(f"Camera {self.camera_index} started successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera"""
        if self.camera:
            self.camera.release()
            self.camera = None
            logger.info("Camera stopped")
    
    def detect_motion(self, frame):
        """Motion detection"""
        try:
            fg_mask = self.background_subtractor.apply(frame)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.motion_threshold:
                    motion_detected = True
                    motion_area += area
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'MOTION DETECTED', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return motion_detected, motion_area, frame
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False, 0, frame
    
    def detect_faces(self, frame):
        """Face detection"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, 'PERSON DETECTED', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            return len(faces) > 0, len(faces), frame
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return False, 0, frame
    
    def trigger_alert(self, alert_type, confidence, frame=None):
        """Trigger incident alert"""
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        incident = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'confidence': confidence,
            'description': f"{alert_type} detected with {confidence:.2f} confidence"
        }
        
        self.incidents.insert(0, incident)
        if len(self.incidents) > 100:
            self.incidents = self.incidents[:100]
        
        # Save screenshot
        if frame is not None and self.save_clips:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.output_dir, f"alert_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)
            incident['screenshot'] = screenshot_path
        
        logger.info(f"ALERT: {alert_type} - Confidence: {confidence:.2f}")
        return incident
    
    def process_frame(self, frame):
        """Process frame with detection"""
        try:
            if frame is None:
                return None
            
            processed_frame = frame.copy()
            
            if self.detection_active:
                # Motion detection
                motion_detected, motion_area, processed_frame = self.detect_motion(processed_frame)
                
                # Face detection
                face_detected, face_count, processed_frame = self.detect_faces(processed_frame)
                
                # Trigger alerts
                if motion_detected and motion_area > self.motion_threshold * (self.motion_sensitivity / 100):
                    confidence = min(motion_area / 50000, 1.0)
                    self.trigger_alert("Motion Detection", confidence, frame)
                
                if face_detected and face_count > 0:
                    confidence = min(face_count * 0.5, 1.0)
                    self.trigger_alert("Person Detection", confidence, frame)
            
            # Add overlays
            self.add_overlays(processed_frame)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame
    
    def add_overlays(self, frame):
        """Add status overlays to frame"""
        try:
            # Status
            status_text = "MONITORING" if self.is_monitoring else "STANDBY"
            status_color = (0, 255, 0) if self.is_monitoring else (0, 0, 255)
            cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Detection status
            detection_text = "ON" if self.detection_active else "OFF"
            detection_color = (0, 255, 0) if self.detection_active else (0, 0, 255)
            cv2.putText(frame, f"Detection: {detection_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, detection_color, 2)
            
            # Camera info
            cv2.putText(frame, f"Camera: {self.camera_index}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Frame count and FPS
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Overlay error: {e}")
    
    def get_frame(self):
        """Get current processed frame"""
        if not self.camera or not self.is_monitoring:
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self.frame_count += 1
                
                # Calculate FPS
                self.fps_count += 1
                current_time = time.time()
                if current_time - self.fps_time >= 1.0:
                    self.current_fps = self.fps_count / (current_time - self.fps_time)
                    self.fps_count = 0
                    self.fps_time = current_time
                
                # Process frame
                processed_frame = self.process_frame(frame)
                self.current_frame = processed_frame
                
                return processed_frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
        
        return None

class IncidentDetectionGUI:
    def __init__(self):
        self.detector = IncidentDetector()
        self.root = tk.Tk()
        self.setup_gui()
        self.update_thread = None
        self.is_updating = False
        
    def setup_gui(self):
        """Setup the main GUI"""
        self.root.title("🛡️ Intelligent Incident Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='white')
        style.configure('Status.TLabel', font=('Arial', 12), background='#2c3e50', foreground='white')
        style.configure('Green.TLabel', font=('Arial', 10, 'bold'), background='#2c3e50', foreground='#27ae60')
        style.configure('Red.TLabel', font=('Arial', 10, 'bold'), background='#2c3e50', foreground='#e74c3c')
        
        self.create_main_layout()
        self.create_video_panel()
        self.create_control_panel()
        self.create_status_panel()
        self.create_settings_panel()
        self.create_incidents_panel()
        
        # Initialize
        self.update_status_display()
        
    def create_main_layout(self):
        """Create main layout structure"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = ttk.Label(title_frame, text="🛡️ Intelligent Incident Detection System", 
                               style='Title.TLabel')
        title_label.pack(expand=True)
        
        # Main content
        self.main_frame = tk.Frame(self.root, bg='#2c3e50')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel (video + controls)
        self.left_panel = tk.Frame(self.main_frame, bg='#34495e', relief='raised', bd=2)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right panel (status + settings + incidents)
        self.right_panel = tk.Frame(self.main_frame, bg='#34495e', width=350, relief='raised', bd=2)
        self.right_panel.pack(side='right', fill='y', padx=(5, 0))
        self.right_panel.pack_propagate(False)
    
    def create_video_panel(self):
        """Create video display panel"""
        video_frame = tk.Frame(self.left_panel, bg='#34495e')
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Video label
        self.video_label = tk.Label(video_frame, bg='black', text='📹\nCamera Feed\nClick Start to begin', 
                                   fg='white', font=('Arial', 16))
        self.video_label.pack(fill='both', expand=True)
    
    def create_control_panel(self):
        """Create control buttons panel"""
        control_frame = tk.Frame(self.left_panel, bg='#34495e', height=80)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Control buttons
        self.start_btn = tk.Button(control_frame, text="🎥 Start Monitoring", 
                                  command=self.start_monitoring, bg='#27ae60', fg='white', 
                                  font=('Arial', 12, 'bold'), relief='raised', bd=3)
        self.start_btn.pack(side='left', padx=5, pady=10, fill='y')
        
        self.stop_btn = tk.Button(control_frame, text="⏹️ Stop Monitoring", 
                                 command=self.stop_monitoring, bg='#e74c3c', fg='white', 
                                 font=('Arial', 12, 'bold'), relief='raised', bd=3)
        self.stop_btn.pack(side='left', padx=5, pady=10, fill='y')
        
        self.toggle_detection_btn = tk.Button(control_frame, text="🔍 Toggle Detection", 
                                            command=self.toggle_detection, bg='#3498db', fg='white', 
                                            font=('Arial', 12, 'bold'), relief='raised', bd=3)
        self.toggle_detection_btn.pack(side='left', padx=5, pady=10, fill='y')
        
        self.test_camera_btn = tk.Button(control_frame, text="📹 Test Camera", 
                                       command=self.test_camera, bg='#9b59b6', fg='white', 
                                       font=('Arial', 12, 'bold'), relief='raised', bd=3)
        self.test_camera_btn.pack(side='left', padx=5, pady=10, fill='y')
    
    def create_status_panel(self):
        """Create status display panel"""
        status_frame = tk.LabelFrame(self.right_panel, text="📊 System Status", 
                                   bg='#34495e', fg='white', font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=10)
        
        # Status indicators
        self.monitoring_status = ttk.Label(status_frame, text="Monitoring: OFFLINE", style='Red.TLabel')
        self.monitoring_status.pack(anchor='w', padx=10, pady=5)
        
        self.detection_status = ttk.Label(status_frame, text="Detection: ACTIVE", style='Green.TLabel')
        self.detection_status.pack(anchor='w', padx=10, pady=5)
        
        self.camera_status = ttk.Label(status_frame, text="Camera: Not Connected", style='Red.TLabel')
        self.camera_status.pack(anchor='w', padx=10, pady=5)
        
        self.fps_status = ttk.Label(status_frame, text="FPS: 0.0", style='Status.TLabel')
        self.fps_status.pack(anchor='w', padx=10, pady=5)
        
        self.frame_status = ttk.Label(status_frame, text="Frames: 0", style='Status.TLabel')
        self.frame_status.pack(anchor='w', padx=10, pady=5)
        
        self.incidents_status = ttk.Label(status_frame, text="Incidents: 0", style='Status.TLabel')
        self.incidents_status.pack(anchor='w', padx=10, pady=5)
    
    def create_settings_panel(self):
        """Create settings panel"""
        settings_frame = tk.LabelFrame(self.right_panel, text="⚙️ Settings", 
                                     bg='#34495e', fg='white', font=('Arial', 12, 'bold'))
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        # Motion sensitivity
        tk.Label(settings_frame, text="Motion Sensitivity:", bg='#34495e', fg='white').pack(anchor='w', padx=10, pady=(10, 0))
        self.motion_sensitivity_var = tk.IntVar(value=50)
        self.motion_scale = tk.Scale(settings_frame, from_=1, to=100, orient='horizontal', 
                                   variable=self.motion_sensitivity_var, bg='#34495e', fg='white',
                                   command=self.update_motion_sensitivity)
        self.motion_scale.pack(fill='x', padx=10, pady=5)
        
        # Camera selection
        tk.Label(settings_frame, text="Camera Index:", bg='#34495e', fg='white').pack(anchor='w', padx=10, pady=(10, 0))
        self.camera_var = tk.IntVar(value=self.detector.camera_index)
        self.camera_spinbox = tk.Spinbox(settings_frame, from_=0, to=5, textvariable=self.camera_var, 
                                       command=self.update_camera_index, width=10)
        self.camera_spinbox.pack(anchor='w', padx=10, pady=5)
        
        # Save clips checkbox
        self.save_clips_var = tk.BooleanVar(value=True)
        self.save_clips_cb = tk.Checkbutton(settings_frame, text="Auto-save incident clips", 
                                          variable=self.save_clips_var, bg='#34495e', fg='white',
                                          command=self.update_save_clips)
        self.save_clips_cb.pack(anchor='w', padx=10, pady=5)
        
        # Settings buttons
        btn_frame = tk.Frame(settings_frame, bg='#34495e')
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(btn_frame, text="💾 Save Settings", command=self.save_settings, 
                 bg='#2980b9', fg='white', font=('Arial', 10)).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="📁 Open Folder", command=self.open_incidents_folder, 
                 bg='#16a085', fg='white', font=('Arial', 10)).pack(side='left', padx=5)
    
    def create_incidents_panel(self):
        """Create incidents log panel"""
        incidents_frame = tk.LabelFrame(self.right_panel, text="🚨 Recent Incidents", 
                                      bg='#34495e', fg='white', font=('Arial', 12, 'bold'))
        incidents_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Incidents list
        self.incidents_text = scrolledtext.ScrolledText(incidents_frame, height=10, width=40, 
                                                       bg='#2c3e50', fg='white', font=('Courier', 9))
        self.incidents_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Clear button
        tk.Button(incidents_frame, text="🧹 Clear Log", command=self.clear_incidents, 
                 bg='#e67e22', fg='white', font=('Arial', 10)).pack(pady=5)
    
    def start_monitoring(self):
        """Start monitoring"""
        try:
            if self.detector.start_camera():
                self.detector.is_monitoring = True
                self.is_updating = True
                
                # Start update thread
                self.update_thread = threading.Thread(target=self.update_video_feed)
                self.update_thread.daemon = True
                self.update_thread.start()
                
                self.update_status_display()
                messagebox.showinfo("Success", "Monitoring started successfully!")
                logger.info("Monitoring started")
            else:
                messagebox.showerror("Error", "Failed to start camera!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        try:
            self.detector.is_monitoring = False
            self.is_updating = False
            
            # Wait for thread to finish
            if self.update_thread:
                self.update_thread.join(timeout=2)
            
            self.detector.stop_camera()
            self.update_status_display()
            
            # Reset video display
            self.video_label.configure(image='', text='📹\nCamera Feed\nClick Start to begin')
            
            messagebox.showinfo("Info", "Monitoring stopped")
            logger.info("Monitoring stopped")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop monitoring: {str(e)}")
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detector.detection_active = not self.detector.detection_active
        self.update_status_display()
        status = "enabled" if self.detector.detection_active else "disabled"
        messagebox.showinfo("Info", f"Detection {status}")
    
    def test_camera(self):
        """Test camera connection"""
        cameras = self.detector.detect_cameras()
        if cameras:
            messagebox.showinfo("Camera Test", f"Found {len(cameras)} working camera(s): {cameras}")
        else:
            messagebox.showerror("Camera Test", "No working cameras found!")
    
    def update_video_feed(self):
        """Update video feed in separate thread"""
        while self.is_updating and self.detector.is_monitoring:
            try:
                frame = self.detector.get_frame()
                if frame is not None:
                    # Convert frame for Tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Resize to fit display
                    display_size = (640, 480)
                    frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update GUI in main thread
                    self.root.after(0, self.update_video_display, frame_tk)
                    
                    # Update status
                    self.root.after(0, self.update_status_display)
                    
                    # Update incidents
                    self.root.after(0, self.update_incidents_display)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Video update error: {e}")
                time.sleep(0.1)
    
    def update_video_display(self, frame_tk):
        """Update video display (called from main thread)"""
        try:
            self.video_label.configure(image=frame_tk, text='')
            self.video_label.image = frame_tk  # Keep a reference
        except Exception as e:
            logger.error(f"Display update error: {e}")
    
    def update_status_display(self):
        """Update status indicators"""
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
            if self.detector.camera:
                self.camera_status.configure(text=f"Camera: Connected ({self.detector.camera_index})", style='Green.TLabel')
            else:
                self.camera_status.configure(text="Camera: Not Connected", style='Red.TLabel')
            
            # FPS and frame count
            self.fps_status.configure(text=f"FPS: {self.detector.current_fps:.1f}")
            self.frame_status.configure(text=f"Frames: {self.detector.frame_count}")
            self.incidents_status.configure(text=f"Incidents: {len(self.detector.incidents)}")
            
        except Exception as e:
            logger.error(f"Status update error: {e}")
    
    def update_incidents_display(self):
        """Update incidents display"""
        try:
            # Check for new incidents
            if hasattr(self, 'last_incident_count'):
                if len(self.detector.incidents) > self.last_incident_count:
                    # New incident detected
                    new_incident = self.detector.incidents[0]
                    incident_text = f"[{datetime.fromisoformat(new_incident['timestamp']).strftime('%H:%M:%S')}] {new_incident['type']} - {new_incident['description']}\n"
                    
                    self.incidents_text.insert(tk.END, incident_text)
                    self.incidents_text.see(tk.END)
            
            self.last_incident_count = len(self.detector.incidents)
            
        except Exception as e:
            logger.error(f"Incidents update error: {e}")
    
    def update_motion_sensitivity(self, value):
        """Update motion sensitivity"""
        self.detector.motion_sensitivity = int(value)
        self.detector.motion_threshold = 5000 * (100 - self.detector.motion_sensitivity) / 100
    
    def update_camera_index(self):
        """Update camera index"""
        self.detector.camera_index = self.camera_var.get()
    
    def update_save_clips(self):
        """Update save clips setting"""
        self.detector.save_clips = self.save_clips_var.get()
    
    def save_settings(self):
        """Save settings to file"""
        try:
            settings = {
                'motion_sensitivity': self.detector.motion_sensitivity,
                'camera_index': self.detector.camera_index,
                'save_clips': self.detector.save_clips,
                'email_alerts': self.detector.email_alerts
            }
            
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo("Success", "Settings saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def open_incidents_folder(self):
        """Open incidents folder"""
        try:
            import subprocess
            subprocess.run(['xdg-open', self.detector.output_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def clear_incidents(self):
        """Clear incidents log"""
        self.incidents_text.delete(1.0, tk.END)
        self.detector.incidents.clear()
        self.update_status_display()
        messagebox.showinfo("Info", "Incidents log cleared")
    
    def on_closing(self):
        """Handle window closing"""
        if self.detector.is_monitoring:
            if messagebox.askokcancel("Quit", "Monitoring is active. Stop monitoring and quit?"):
                self.stop_monitoring()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize last incident count
        self.last_incident_count = 0
        
        # Start GUI main loop
        self.root.mainloop()

if __name__ == '__main__':
    try:
        print("🛡️ Starting Intelligent Incident Detection System (Tkinter GUI)")
        print("📱 GUI interface will open shortly...")
        print("🎥 Make sure your webcam is connected")
        print("📁 Incident recordings will be saved to: ./incidents/")
        print("\n🚀 Starting GUI...")
        
        app = IncidentDetectionGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
    finally:
        print("✅ Application closed")