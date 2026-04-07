#!/usr/bin/env python3
"""
Standalone Face Recognition Script

Takes an image file as input and outputs the recognized person's name.

Usage:
    python face_standalone.py <image_path>
    
Example:
    python face_standalone.py test_image.jpg
"""

import sys
import os
import cv2
from face_recognition import FaceRecognitionHandler
from logger_config import setup_logger
from config import FACE_MIN_SIZE_RT

FACE_DEVICE_ID = 1

logger = setup_logger(__name__, "face_standalone.log")

def recognize_image(image_path):
    """
    Main function to recognize faces in an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        list: List of recognized names (or "Unknown" for unrecognized faces)
              Returns empty list if no faces detected or on error
    """
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}")
        return []
    
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Failed to load image")
        return []
    
    logger.info("Initializing face recognition model...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FACE_DEVICE_ID) if FACE_DEVICE_ID != -1 else ''
    
    face_handler = FaceRecognitionHandler()
    if not face_handler.load_face_recognition(device_id=FACE_DEVICE_ID):
        logger.error("Failed to load face recognition model!")
        return []
    
    logger.info(f"Face recognition loaded. Known people: {len(face_handler.face_db)}")
    
    logger.info("Detecting faces...")
    try:
        faces = face_handler.face_app.get(image)
    except Exception as e:
        logger.error(f"Error in face detection: {e}", exc_info=True)
        return []
    
    if len(faces) == 0:
        logger.info("No faces detected in the image")
        return []
    
    logger.info(f"Found {len(faces)} face(s)")
    logger.info("-" * 50)
    
    recognized_names = []
    
    for idx, face in enumerate(faces, 1):
        face_bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = face_bbox
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        if face_width < FACE_MIN_SIZE_RT[0] or face_height < FACE_MIN_SIZE_RT[1]:
            logger.info(f"Face {idx}: Skipped (too small: {face_width}x{face_height} < {FACE_MIN_SIZE_RT})")
            continue
        
        try:
            embedding = face.embedding.flatten() if hasattr(face.embedding, 'flatten') else face.embedding
            name = face_handler.identify_face(embedding)
        except Exception as e:
            logger.error(f"Error identifying face: {e}", exc_info=True)
            name = None
        
        if name:
            logger.info(f"Face {idx}: {name}")
            recognized_names.append(name)
        else:
            logger.info(f"Face {idx}: Unknown")
            recognized_names.append("Unknown")
        
        logger.info(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
        logger.info(f"  Detection confidence: {face.det_score:.3f}")
    
    return recognized_names

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("Usage: python face_standalone.py <image_path>")
        logger.error("Example: python face_standalone.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    names = recognize_image(image_path)
    pp = 1
