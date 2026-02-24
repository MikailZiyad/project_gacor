from deepface import DeepFace
import numpy as np
import cv2
from pathlib import Path
import json
import os
from config.settings import FACE_RECOGNITION_THRESHOLD, FACE_RECOGNITION_MODEL, MODELS_PATH
from models.database import DatabaseManager
from config.constants import FACE_QUALITY_REQUIREMENTS

class FaceRecognizer:
    def __init__(self):
        self.threshold = FACE_RECOGNITION_THRESHOLD
        self.model_name = FACE_RECOGNITION_MODEL
        self.db = DatabaseManager()
        self.last_error = None
        try:
            Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
            os.environ["DEEPFACE_HOME"] = str(MODELS_PATH)
        except Exception:
            pass
    
    def extract_embedding(self, face_image):
        self.last_error = None
        if face_image is None or face_image.size == 0:
            self.last_error = "Empty face image"
            return None
        img = face_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        attempts = [
            {"model": self.model_name, "backend": "skip", "align": False},
            {"model": "Facenet512", "backend": "skip", "align": False},
        ]
        errors = []
        for at in attempts:
            try:
                result = DeepFace.represent(
                    img,
                    model_name=at["model"],
                    detector_backend=at["backend"],
                    enforce_detection=False,
                    align=at["align"]
                )
                if result and len(result) > 0 and "embedding" in result[0]:
                    return result[0]["embedding"]
            except Exception as e:
                errors.append(f"{at['model']}: {str(e)}")
                continue
        joined = "; ".join(errors) if errors else "No embeddings extracted"
        if "weights" in joined or "download" in joined or "HTTP" in joined or "urlopen" in joined:
            joined = f"DeepFace weights not available yet. Connect internet once to download into {MODELS_PATH}\\weights"
        self.last_error = joined
        print(f"Error extracting embedding: {self.last_error}")
        return None
    
    def register_face(self, employee_id, face_samples):
        embeddings = []
        
        for i, face_sample in enumerate(face_samples):
            embedding = self.extract_embedding(face_sample)
            if embedding:
                embeddings.append(embedding)
                
                # Save face image
                face_path = Path(f"data/employees/{employee_id}_sample_{i}.jpg")
                cv2.imwrite(str(face_path), face_sample)
                
                # Save embedding to database
                self.db.add_face_embedding(employee_id, embedding, str(face_path))
        
        if embeddings:
            print(f"✅ Berhasil mendaftarkan {len(embeddings)} sample wajah untuk {employee_id}")
            return True
        else:
            print(f"❌ Gagal mendapatkan embedding dari wajah: {self.last_error or 'unknown error'}")
            return False
    
    def recognize_face(self, face_image):
        """
        Recognize a face and return the closest match from database
        """
        try:
            # Extract embedding from the input face
            input_embedding = self.extract_embedding(face_image)
            
            if input_embedding is None:
                return {
                    'recognized': False,
                    'employee_id': None,
                    'confidence': 0.0,
                    'message': 'Gagal mengekstrak fitur wajah'
                }
            
            # Get all employees with face embeddings
            employees = self.db.get_all_employees()
            
            best_match = None
            best_confidence = 0.0
            
            for employee in employees:
                employee_id = employee[1]  # employee_id column
                
                # Get embeddings for this employee
                embeddings = self.db.get_face_embeddings(employee_id)
                
                if not embeddings:
                    continue
                
                # Calculate average similarity with all stored embeddings
                similarities = []
                for stored_embedding in embeddings:
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(input_embedding, stored_embedding)
                    similarities.append(similarity)
                
                # Use the maximum similarity (best match among samples)
                max_similarity = max(similarities)
                
                if max_similarity > best_confidence:
                    best_confidence = max_similarity
                    best_match = employee_id
            
            # Check if confidence is above threshold
            if best_confidence >= self.threshold:
                return {
                    'recognized': True,
                    'employee_id': best_match,
                    'confidence': best_confidence,
                    'message': f'Wajah dikenali sebagai {best_match} dengan confidence {best_confidence:.2f}'
                }
            else:
                return {
                    'recognized': False,
                    'employee_id': None,
                    'confidence': best_confidence,
                    'message': f'Wajah tidak dikenali (confidence: {best_confidence:.2f})'
                }
                
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return {
                'recognized': False,
                'employee_id': None,
                'confidence': 0.0,
                'message': f'Error: {str(e)}'
            }
    
    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def verify_face(self, face_image):
        current_embedding = self.extract_embedding(face_image)
        if current_embedding is None:
            return None, 0.0
        
        # Get all employees with face embeddings
        employees = self.db.get_all_employees()
        
        best_match = None
        best_confidence = 0.0
        
        for employee in employees:
            employee_id = employee[1]  # employee_id column
            
            # Get stored embeddings for this employee
            stored_embeddings = self.db.get_face_embeddings(employee_id)
            
            if not stored_embeddings:
                continue
            
            # Compare with each stored embedding
            for stored_embedding in stored_embeddings:
                try:
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(current_embedding, stored_embedding)
                    confidence = similarity * 100
                    
                    if similarity > best_confidence and similarity > self.threshold:
                        best_match = employee_id
                        best_confidence = similarity
                        
                except Exception as e:
                    print(f"Error comparing embeddings: {e}")
                    continue
        
        return best_match, best_confidence
    
    def cosine_similarity_old(self, embedding1, embedding2):
        """Old version - kept for compatibility"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def validate_face_quality(self, face_image):
        if face_image is None or face_image.size == 0:
            return False, "No face detected"
        h, w = face_image.shape[:2]
        area = h * w
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        min_b = FACE_QUALITY_REQUIREMENTS.get("min_brightness", 40)
        max_b = FACE_QUALITY_REQUIREMENTS.get("max_brightness", 210)
        min_blur = FACE_QUALITY_REQUIREMENTS.get("min_blur_score", 60)
        min_area = FACE_QUALITY_REQUIREMENTS.get("min_face_size", 8000)
        if area < min_area:
            return False, "Face too small"
        if blur_score < min_blur:
            return False, "Face too blurry"
        if brightness < min_b or brightness > max_b:
            return False, "Poor lighting"
        return True, "Good quality"
    
    def get_face_quality_score(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        h, w = face_image.shape[:2]
        area = h * w
        min_blur = float(FACE_QUALITY_REQUIREMENTS.get("min_blur_score", 60))
        min_b = float(FACE_QUALITY_REQUIREMENTS.get("min_brightness", 40))
        max_b = float(FACE_QUALITY_REQUIREMENTS.get("max_brightness", 210))
        min_area = float(FACE_QUALITY_REQUIREMENTS.get("min_face_size", 8000))
        blur_score = max(0.0, min(100.0, (blur_val / max(1.0, min_blur)) * 80.0))
        target_b = (min_b + max_b) / 2.0
        brightness_score = max(0.0, 100.0 - abs(brightness - target_b) * 0.6)
        size_score = max(0.0, min(100.0, (area / max(1.0, min_area)) * 80.0))
        quality_score = (blur_score + brightness_score + size_score) / 3.0
        is_valid, message = self.validate_face_quality(face_image)
        if not is_valid:
            return quality_score, f"Poor face quality: {message}"
        return quality_score, "Good quality"
