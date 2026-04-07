#!/usr/bin/env python3
"""
Face Registration Script
Register new faces by capturing images and generating embeddings
"""

import cv2
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add the face_standalone directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_recognition import FaceRecognitionHandler
from config import FACE_DB_PATH, FACE_MIN_SIZE_RT

class FaceRegistration:
    def __init__(self):
        self.face_handler = FaceRecognitionHandler()
        self.face_handler.load_face_recognition(device_id=0)  # Use GPU
        
    def register_from_images(self, image_paths, person_name):
        """
        Register a new person from multiple images (recommended for better accuracy)
        
        Args:
            image_paths: List of paths to image files
            person_name: Name to register
        """
        successful_registrations = 0
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print("Failed to load image")
                continue
            
            # Detect faces
            faces = self.face_handler.face_app.get(image)
            
            if len(faces) == 0:
                print("No faces detected in image")
                continue
            
            if len(faces) > 1:
                print(f"Multiple faces detected ({len(faces)}). Skipping this image.")
                continue
            
            # Get the face
            face = faces[0]
            face_bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = face_bbox
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            if face_width < FACE_MIN_SIZE_RT[0] or face_height < FACE_MIN_SIZE_RT[1]:
                print(f"Face too small: {face_width}x{face_height}")
                continue
            
            # Save face crop
            face_crop_path = self.face_handler.save_face_image(face_bbox, image, person_name)
            if not face_crop_path:
                print("Failed to save face crop")
                continue
            
            # Add embedding to database
            embedding = face.embedding.flatten() if hasattr(face.embedding, 'flatten') else face.embedding
            if self.face_handler.add_embedding_to_database(person_name, embedding):
                successful_registrations += 1
                print(f"Successfully registered face {successful_registrations} for {person_name}")
            else:
                print("Failed to add embedding to database")
        
        if successful_registrations > 0:
            print(f"Successfully registered {person_name} with {successful_registrations} face embeddings")
            # Reload database
            self.face_handler.reload_face_database()
            return True
        else:
            print(f"Failed to register any faces for {person_name}")
            return False
    
    def register_from_image(self, image_path, person_name):
        """
        Register a new person from a single image
        
        Args:
            image_path: Path to image file
            person_name: Name to register
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
        
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image")
            return False
        
        # Detect faces
        faces = self.face_handler.face_app.get(image)
        
        if len(faces) == 0:
            print("No faces detected in image")
            return False
        
        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}). Please use image with single face.")
            return False
        
        # Get the face
        face = faces[0]
        face_bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = face_bbox
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        if face_width < FACE_MIN_SIZE_RT[0] or face_height < FACE_MIN_SIZE_RT[1]:
            print(f"Face too small: {face_width}x{face_height}")
            return False
        
        # Save face crop
        face_crop_path = self.face_handler.save_face_image(face_bbox, image, person_name)
        if not face_crop_path:
            print("Failed to save face crop")
            return False
        
        # Add embedding to database
        embedding = face.embedding.flatten() if hasattr(face.embedding, 'flatten') else face.embedding
        if self.face_handler.add_embedding_to_database(person_name, embedding):
            print(f"Successfully registered {person_name}")
            print(f"Face crop saved: {face_crop_path}")
            
            # Reload database
            self.face_handler.reload_face_database()
            return True
        else:
            print("Failed to add embedding to database")
            return False
    
    def register_from_camera(self, person_name):
        """
        Register a new person from live camera feed
        
        Args:
            person_name: Name to register
        """
        print(f"Registering {person_name} from camera...")
        print("Press SPACE to capture, ESC to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            return False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Detect faces
            faces = self.face_handler.face_app.get(frame)
            
            # Draw face boxes
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow('Face Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if len(faces) == 1:
                    face = faces[0]
                    face_bbox = face.bbox.astype(int)
                    
                    face_width = face_bbox[2] - face_bbox[0]
                    face_height = face_bbox[3] - face_bbox[1]
                    
                    if face_width >= FACE_MIN_SIZE_RT[0] and face_height >= FACE_MIN_SIZE_RT[1]:
                        # Save face crop
                        face_crop_path = self.face_handler.save_face_image(face_bbox, frame, person_name)
                        if face_crop_path:
                            # Add embedding
                            embedding = face.embedding.flatten() if hasattr(face.embedding, 'flatten') else face.embedding
                            if self.face_handler.add_embedding_to_database(person_name, embedding):
                                print(f"Successfully registered {person_name}")
                                self.face_handler.reload_face_database()
                                break
                        else:
                            print("Failed to save face crop")
                    else:
                        print(f"Face too small: {face_width}x{face_height}")
                elif len(faces) == 0:
                    print("No face detected")
                else:
                    print(f"Multiple faces detected ({len(faces)})")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def list_registered_faces(self):
        """List all registered faces"""
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, 'r') as f:
                data = json.load(f)
                print("Registered faces:")
                for name, embeddings in data['embeddings'].items():
                    print(f"  - {name} ({len(embeddings)} embeddings)")
        else:
            print("No face database found")
    
    def remove_face(self, person_name):
        """Remove a person from the database"""
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, 'r') as f:
                data = json.load(f)
            
            if person_name in data['embeddings']:
                del data['embeddings'][person_name]
                
                with open(FACE_DB_PATH, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Removed {person_name} from database")
                
                # Remove face images
                face_data_dir = Path("static/face_data") / person_name
                if face_data_dir.exists():
                    import shutil
                    shutil.rmtree(face_data_dir)
                    print(f"Removed face images for {person_name}")
                
                self.face_handler.reload_face_database()
                return True
            else:
                print(f"{person_name} not found in database")
                return False
        else:
            print("No face database found")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Registration Tool")
    parser.add_argument("action", choices=["list", "add", "add-multi", "remove", "camera"], 
                       help="Action to perform")
    parser.add_argument("--name", help="Person name (for add/remove/camera)")
    parser.add_argument("--image", help="Image path (for add)")
    parser.add_argument("--images", nargs='+', help="Multiple image paths (for add-multi)")
    
    args = parser.parse_args()
    
    reg = FaceRegistration()
    
    if args.action == "list":
        reg.list_registered_faces()
    
    elif args.action == "add":
        if not args.name or not args.image:
            print("Error: --name and --image required for add")
            return
        reg.register_from_image(args.image, args.name)
    
    elif args.action == "add-multi":
        if not args.name or not args.images:
            print("Error: --name and --images required for add-multi")
            return
        reg.register_from_images(args.images, args.name)
    
    elif args.action == "camera":
        if not args.name:
            print("Error: --name required for camera")
            return
        reg.register_from_camera(args.name)
    
    elif args.action == "remove":
        if not args.name:
            print("Error: --name required for remove")
            return
        reg.remove_face(args.name)

if __name__ == '__main__':
    main()
