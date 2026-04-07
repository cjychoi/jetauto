import os
import json
import numpy as np
import cv2
from pathlib import Path
from insightface.app import FaceAnalysis
from config import FACE_DB_PATH, FACE_RECOGNITION_THRESHOLD, FACE_MIN_SIZE_RT, FACE_DEVICE_ID
from sync_persons import sync_persons_to_node_sets
from logger_config import setup_logger

logger = setup_logger(__name__, "face_recognition.log")

class FaceRecognitionHandler:
    def __init__(self):
        self.face_app = None
        self.face_db = {}
        self.track_id_to_name = {}
        self.auto_register_unknown = True
        self.registered_embeddings = set()
    
    def load_face_recognition(self, device_id=None):
        try:
            if device_id is None:
                device_id = FACE_DEVICE_ID
            
            if device_id == -1:
                providers = ['CPUExecutionProvider']
            else:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.face_app.prepare(ctx_id=device_id, det_size=(640, 640))
            
            if os.path.exists(FACE_DB_PATH):
                with open(FACE_DB_PATH, 'r') as f:
                    data = json.load(f)
                    self.face_db = {name: [np.array(emb) for emb in embs] 
                                   for name, embs in data.get('embeddings', {}).items()}
                logger.info(f"Face database loaded: {list(self.face_db.keys())}")
                sync_persons_to_node_sets()
            else:
                logger.warning(f"Face database not found at {FACE_DB_PATH}")
            
            logger.info("InsightFace loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}", exc_info=True)
            return False
    
    def reload_face_database(self):
        try:
            if os.path.exists(FACE_DB_PATH):
                with open(FACE_DB_PATH, 'r') as f:
                    data = json.load(f)
                    self.face_db = {name: [np.array(emb) for emb in embs] 
                                   for name, embs in data.get('embeddings', {}).items()}
                self.track_id_to_name.clear()
                logger.info(f"Face database reloaded: {list(self.face_db.keys())}")
                sync_persons_to_node_sets()
                return True, list(self.face_db.keys())
            else:
                logger.warning(f"Face database not found at {FACE_DB_PATH}")
                return False, []
        except Exception as e:
            logger.error(f"Failed to reload face database: {e}", exc_info=True)
            return False, []
    
    def get_next_person_number(self):
        """
        Find the highest candidate_x number in the database and return next number.
        """
        max_person = 0
        for name in self.face_db.keys():
            if name.startswith("candidate_"):
                try:
                    num = int(name.split("_")[1])
                    max_person = max(max_person, num)
                except (ValueError, IndexError):
                    continue
        return max_person + 1
    
    def save_face_image(self, face_bbox, frame, person_name, padding=0.5):
        """
        Save the cropped face image to static/face_data/candidate_name/
        """
        try:
            face_data_dir = Path("static/face_data")
            person_dir = face_data_dir / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            
            x1, y1, x2, y2 = face_bbox
            h, w = frame.shape[:2]
            
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            face_crop = frame[y1:y2, x1:x2]
            
            existing_images = list(person_dir.glob("*.jpg"))
            image_num = len(existing_images) + 1
            image_path = person_dir / f"image_{image_num}.jpg"
            
            cv2.imwrite(str(image_path), face_crop)
            return image_path
        except Exception as e:
            logger.error(f"Error saving face image: {e}", exc_info=True)
            return None
    
    def add_embedding_to_database(self, person_name, embedding):
        """
        Add a new person's embedding to the face database JSON file.
        """
        try:
            if os.path.exists(FACE_DB_PATH):
                with open(FACE_DB_PATH, 'r') as f:
                    data = json.load(f)
            else:
                data = {"embeddings": {}}
            
            if person_name not in data["embeddings"]:
                data["embeddings"][person_name] = []
            
            data["embeddings"][person_name].append(embedding.tolist())
            
            with open(FACE_DB_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error updating database: {e}", exc_info=True)
            return False
    
    def identify_face(self, face_embedding):
        if not self.face_db:
            return None
        
        # Ensure face_embedding is 1D array
        face_embedding = face_embedding.flatten() if hasattr(face_embedding, 'flatten') else face_embedding
        
        best_match = None
        best_score = -1
        
        try:
            for name, embeddings in self.face_db.items():
                for db_emb in embeddings:
                    # Ensure db_emb is also 1D array to avoid shape mismatch
                    db_emb = db_emb.flatten() if hasattr(db_emb, 'flatten') else db_emb
                    
                    score = np.dot(face_embedding, db_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(db_emb))
                    if score > best_score:
                        best_score = score
                        best_match = name
        except Exception as e:
            logger.error(f"Error in face identification: {e}", exc_info=True)
            return None
        
        if best_score >= FACE_RECOGNITION_THRESHOLD:
            return best_match
        return None
    
    def recognize_faces_in_boxes(self, frame, person_boxes):  # Yu: This recognizes and adds unknown face crops (if not recognizable)
        if not self.face_app:
            return {}
        
        identifications = {}
        
        try:
            faces = self.face_app.get(frame)
            
            for face in faces:
                face_bbox = face.bbox.astype(int)
                face_center = ((face_bbox[0] + face_bbox[2]) // 2, (face_bbox[1] + face_bbox[3]) // 2)
                
                for track_id, person_box in person_boxes.items():
                    px1, py1, px2, py2 = person_box
                    if px1 <= face_center[0] <= px2 and py1 <= face_center[1] <= py2:
                        name = self.identify_face(face.embedding)
                        
                        if name:
                            identifications[track_id] = name
                            self.track_id_to_name[track_id] = name
                        elif self.auto_register_unknown and track_id not in self.track_id_to_name:
                            face_width = face_bbox[2] - face_bbox[0]
                            face_height = face_bbox[3] - face_bbox[1]
                            
                            if face_width >= FACE_MIN_SIZE_RT[0] and face_height >= FACE_MIN_SIZE_RT[1]:
                                embedding_hash = hash(face.embedding.tobytes())
                                if embedding_hash not in self.registered_embeddings:
                                    next_person_num = self.get_next_person_number()
                                    person_name = f"candidate_{next_person_num}"
                                    
                                    image_path = self.save_face_image(face_bbox, frame, person_name)
                                    if image_path and self.add_embedding_to_database(person_name, face.embedding):
                                        logger.info(f"Auto-registered unknown person as '{person_name}' (face size: {face_width}x{face_height})")
                                        self.face_db[person_name] = [face.embedding]
                                        self.track_id_to_name[track_id] = person_name
                                        identifications[track_id] = person_name
                                        self.registered_embeddings.add(embedding_hash)
                                        sync_persons_to_node_sets()
                            else:
                                logger.debug(f"Skipped auto-registration: face too small ({face_width}x{face_height} < {FACE_MIN_SIZE_RT})")
                        break
        except Exception as e:
            logger.error(f"Error in recognize_faces_in_boxes: {e}", exc_info=True)
        
        return identifications
