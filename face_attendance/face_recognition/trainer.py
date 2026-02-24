import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FaceRecognizer
from models.employee import Employee
from models.database import DatabaseManager
from config.constants import FACE_POSES, SAMPLES_PER_POSE, FACE_QUALITY_REQUIREMENTS
from config.settings import FACE_IMAGES_PATH, FACE_CONFIDENCE_THRESHOLD

class FaceTrainer:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.db = DatabaseManager()
        self.face_samples = []
        self.current_pose_index = 0
        self.samples_collected = 0
        self.employee_id = None
        self.employee_name = None
        
    def start_registration(self, employee_id, employee_name):
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.face_samples = []
        self.current_pose_index = 0
        self.samples_collected = 0
        
        # Create employee folder
        employee_folder = Path(FACE_IMAGES_PATH) / employee_id
        employee_folder.mkdir(exist_ok=True)
        
        print(f"📝 Memulai registrasi untuk {employee_name} ({employee_id})")
        print(f"📸 Target: {len(FACE_POSES)} pose × {SAMPLES_PER_POSE} sample = {len(FACE_POSES) * SAMPLES_PER_POSE} foto")
        
        return True
    
    def capture_frame(self, frame):
        if self.current_pose_index >= len(FACE_POSES):
            return False, "Registration complete"
        
        # Detect faces
        faces = self.detector.detect_faces(frame)
        
        if len(faces) == 0:
            return False, "No face detected"
        
        if len(faces) > 1:
            return False, "Multiple faces detected"
        
        # Extract face
        face_data = faces[0]
        face_image, bbox = self.detector.extract_face(frame, face_data['bbox'])
        
        # Validate face quality
        quality_score, quality_message = self.recognizer.get_face_quality_score(face_image)
        
        if quality_score < 60:
            return False, f"Poor face quality: {quality_message}"
        
        # Add to samples
        self.face_samples.append(face_image.copy())
        self.samples_collected += 1
        
        # Save sample image
        pose_name = FACE_POSES[self.current_pose_index].replace(" ", "_").lower()
        if FACE_IMAGES_PATH and self.employee_id:
            sample_path = Path(FACE_IMAGES_PATH) / self.employee_id / f"{pose_name}_sample_{self.samples_collected % SAMPLES_PER_POSE + 1}.jpg"
            cv2.imwrite(str(sample_path), face_image)
        
        # Check if we have enough samples for current pose
        if self.samples_collected % SAMPLES_PER_POSE == 0:
            self.current_pose_index += 1
        
        current_pose = FACE_POSES[min(self.current_pose_index, len(FACE_POSES) - 1)]
        progress = (self.samples_collected / (len(FACE_POSES) * SAMPLES_PER_POSE)) * 100
        
        status_msg = f"Pose: {current_pose} | Progress: {progress:.1f}% | Samples: {self.samples_collected}"
        
        return True, status_msg
    
    def is_registration_complete(self):
        return self.samples_collected >= len(FACE_POSES) * SAMPLES_PER_POSE
    
    def get_current_instruction(self):
        if self.current_pose_index < len(FACE_POSES):
            return FACE_POSES[self.current_pose_index]
        return "Registration complete"
    
    def get_progress(self):
        total_samples = len(FACE_POSES) * SAMPLES_PER_POSE
        return (self.samples_collected / total_samples) * 100 if total_samples > 0 else 0
    
    def complete_registration(self):
        if not self.is_registration_complete():
            return False, "Not enough samples collected"
        
        if len(self.face_samples) < 10:
            return False, "Too few valid samples"
        
        try:
            # Register employee in database
            success, message = self.db.add_employee(self.employee_id, self.employee_name)
            if not success:
                return False, message
            
            # Train face recognition with collected samples
            training_success = self.recognizer.register_face(self.employee_id, self.face_samples)
            
            if training_success:
                print(f"✅ Registrasi berhasil untuk {self.employee_name}")
                print(f"📊 Total samples yang digunakan: {len(self.face_samples)}")
                return True, "Registration successful"
            else:
                detail = self.recognizer.last_error or "no embeddings extracted"
                return False, f"Face training failed: {detail}"
                
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def reset_registration(self):
        self.face_samples = []
        self.current_pose_index = 0
        self.samples_collected = 0
        self.employee_id = None
        self.employee_name = None
        
        print("🔄 Registrasi direset")
    
    def draw_registration_ui(self, frame, status_msg=""):
        faces = self.detector.detect_faces(frame)
        h, w = frame.shape[:2]
        roi_w = int(w * 0.5)
        roi_h = int(h * 0.6)
        roi_x1 = (w - roi_w) // 2
        roi_y1 = (h - roi_h) // 2
        roi_x2 = roi_x1 + roi_w
        roi_y2 = roi_y1 + roi_h
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 3)
        for face_data in faces:
            x1, y1, x2, y2 = face_data['bbox']
            conf = face_data['confidence']
            inside = x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2
            color = (0, 255, 0) if (inside and conf >= FACE_CONFIDENCE_THRESHOLD) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            try:
                face_img, _ = self.detector.extract_face(frame, (x1, y1, x2, y2))
                if face_img is not None:
                    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
                    bright = gray.mean()
                    text = f"{conf:.2f} | bl:{blur_val:.0f} br:{bright:.0f}"
                else:
                    text = f"{conf:.2f}"
            except Exception:
                text = f"{conf:.2f}"
            cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw instruction text
        if self.current_pose_index < len(FACE_POSES):
            instruction = self.get_current_instruction()
            progress = self.get_progress()
            
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
            
            cv2.putText(frame, f"Instruksi: {instruction}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Progress: {progress:.1f}%", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Samples: {self.samples_collected}/{len(FACE_POSES) * SAMPLES_PER_POSE}", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if status_msg:
                cv2.putText(frame, f"Status: {status_msg}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
