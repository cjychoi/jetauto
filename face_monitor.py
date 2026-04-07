#!/usr/bin/env python3
"""
Face Recognition Folder Monitor

Monitors a folder for new images and performs face recognition on them.
Loads the face recognition model once and keeps it in memory.
Outputs results to a file with timestamp and recognized names.

Usage:
    python face_monitor.py <folder_path> [--output <output_file>] [--interval <seconds>]
    
Example:
    python face_monitor.py ./images --output results.txt --interval 1
"""

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import cv2
from face_recognition import FaceRecognitionHandler
from logger_config import setup_logger
from config import FACE_MIN_SIZE_RT

logger = setup_logger(__name__, "face_monitor.log")

class FaceMonitor:
    def __init__(self, folder_path, output_file="face_recognition_results.txt", device_id=1):
        """
        Initialize the face monitor
        
        Args:
            folder_path: Path to folder to monitor
            output_file: Path to output results file
            device_id: GPU device ID (-1 for CPU)
        """
        self.folder_path = Path(folder_path)
        self.output_file = Path(output_file)
        self.device_id = device_id
        self.processed_files = set()
        self.face_handler = None
        
        if not self.folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def initialize_model(self):
        """Load the face recognition model once"""
        logger.info("Initializing face recognition model...")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id) if self.device_id != -1 else ''
        
        self.face_handler = FaceRecognitionHandler()
        if not self.face_handler.load_face_recognition(device_id=self.device_id):
            logger.error("Failed to load face recognition model!")
            return False
        
        logger.info(f"Face recognition loaded. Known people: {len(self.face_handler.face_db)}")
        return True
    
    def recognize_faces_in_image(self, image_path, max_retries=3, retry_delay=0.5):
        """
        Recognize faces in a single image
        
        Args:
            image_path: Path to image file
            max_retries: Maximum number of retry attempts for incomplete files
            retry_delay: Delay in seconds between retries
            
        Returns:
            list: List of recognized names (or "Unknown" for unrecognized faces)
        """
        image = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    break
                last_error = "cv2.imread returned None"
            except Exception as e:
                last_error = str(e)
                if "Premature end" in str(e) or "JPEG" in str(e):
                    if attempt < max_retries - 1:
                        logger.debug(f"Image not ready, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries}): {image_path.name}")
                        time.sleep(retry_delay)
                        continue
                raise
        
        if image is None:
            logger.error(f"Failed to load image after {max_retries} attempts: {image_path} - {last_error}")
            return []
        
        try:
            
            faces = self.face_handler.face_app.get(image)
            
            if len(faces) == 0:
                logger.info(f"No faces detected in {image_path.name}")
                return []
            
            recognized_names = []
            
            for idx, face in enumerate(faces, 1):
                face_bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = face_bbox
                
                face_width = x2 - x1
                face_height = y2 - y1
                
                if face_width < FACE_MIN_SIZE_RT[0] or face_height < FACE_MIN_SIZE_RT[1]:
                    logger.debug(f"Face {idx} in {image_path.name}: Skipped (too small: {face_width}x{face_height})")
                    continue
                
                try:
                    embedding = face.embedding.flatten() if hasattr(face.embedding, 'flatten') else face.embedding
                    name = self.face_handler.identify_face(embedding)
                except Exception as e:
                    logger.error(f"Error identifying face in {image_path.name}: {e}", exc_info=True)
                    name = None
                
                if name:
                    recognized_names.append(name)
                    logger.info(f"Face {idx} in {image_path.name}: {name}")
                else:
                    recognized_names.append("Unknown")
                    logger.info(f"Face {idx} in {image_path.name}: Unknown")
            
            return recognized_names
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            return []
    
    def write_result(self, image_path, names):
        """
        Write recognition result to output file
        
        Args:
            image_path: Path to the image
            names: List of recognized names
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        names_str = ", ".join(names) if names else "No faces detected"
        
        result_line = f"{timestamp} | {image_path.name} | {names_str}\n"
        
        with open(self.output_file, 'a') as f:
            f.write(result_line)
        
        logger.info(f"Result written: {image_path.name} -> {names_str}")
    
    def is_file_stable(self, file_path, wait_time=0.1):
        """
        Check if file is stable (not being written to)
        
        Args:
            file_path: Path to file
            wait_time: Time to wait between size checks
            
        Returns:
            bool: True if file size is stable
        """
        try:
            size1 = file_path.stat().st_size
            time.sleep(wait_time)
            size2 = file_path.stat().st_size
            return size1 == size2 and size1 > 0
        except Exception:
            return False
    
    def get_new_images(self):
        """
        Get list of new images in the folder that haven't been processed
        
        Returns:
            list: List of new image paths
        """
        new_images = []
        
        for file_path in self.folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                if str(file_path) not in self.processed_files:
                    if self.is_file_stable(file_path):
                        new_images.append(file_path)
                    else:
                        logger.debug(f"File still being written, skipping for now: {file_path.name}")
        
        return sorted(new_images)
    
    def process_image(self, image_path):
        """
        Process a single image: recognize faces and write results
        
        Args:
            image_path: Path to image file
        """
        logger.info(f"Processing: {image_path.name}")
        
        names = self.recognize_faces_in_image(image_path)
        self.write_result(image_path, names)
        self.processed_files.add(str(image_path))
    
    def monitor(self, interval=1.0):
        """
        Monitor the folder and process new images
        
        Args:
            interval: Check interval in seconds
        """
        logger.info(f"Starting to monitor folder: {self.folder_path}")
        logger.info(f"Results will be written to: {self.output_file}")
        logger.info(f"Check interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop")
        
        with open(self.output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Monitoring started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
        
        try:
            while True:
                new_images = self.get_new_images()
                
                if new_images:
                    logger.info(f"Found {len(new_images)} new image(s)")
                    for image_path in new_images:
                        self.process_image(image_path)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped by user (Ctrl+C)")
            with open(self.output_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Monitoring stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
            logger.info(f"Total images processed: {len(self.processed_files)}")


def main():
    parser = argparse.ArgumentParser(description="Monitor a folder for new images and perform face recognition")
    parser.add_argument("folder", help="Path to folder to monitor")
    parser.add_argument("--output", "-o", default="face_recognition_results.txt", 
                        help="Output file for results (default: face_recognition_results.txt)")
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                        help="Check interval in seconds (default: 1.0)")
    parser.add_argument("--device", "-d", type=int, default=1,
                        help="GPU device ID, -1 for CPU (default: 1)")
    
    args = parser.parse_args()
    
    try:
        monitor = FaceMonitor(args.folder, args.output, args.device)
        
        if not monitor.initialize_model():
            logger.error("Failed to initialize face recognition model")
            sys.exit(1)
        
        monitor.monitor(args.interval)
        
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
