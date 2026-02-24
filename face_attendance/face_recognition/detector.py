import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from config.settings import YOLO_MODEL_PATH, FACE_CONFIDENCE_THRESHOLD

class FaceDetector:
    def __init__(self):
        self.model_path = YOLO_MODEL_PATH
        self.confidence_threshold = FACE_CONFIDENCE_THRESHOLD
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"Model YOLO berhasil dimuat dari: {self.model_path}")
        except Exception as e:
            print(f"Error memuat model YOLO: {e}")
            raise e
    
    def detect_faces(self, frame):
        if self.model is None:
            return []
        
        results = self.model(frame)
        
        faces = []
        best_candidate = None
        best_conf = 0.0
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if conf > best_conf:
                        best_conf = conf
                        best_candidate = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class': cls
                        }
                    
                    if conf >= self.confidence_threshold:
                        face_data = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class': cls
                        }
                        faces.append(face_data)
        
        # Fallback: jika tidak ada yang melewati threshold, namun ada kandidat cukup baik
        if not faces and best_candidate and best_conf >= max(0.3, self.confidence_threshold * 0.7):
            faces.append(best_candidate)
        
        return faces
    
    def extract_face(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        face = frame[y1:y2, x1:x2]
        return face, (x1, y1, x2, y2)
    
    def draw_face_boxes(self, frame, faces):
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            conf = face['confidence']
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Face: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
