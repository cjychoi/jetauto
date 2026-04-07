#!/usr/bin/env python3
"""
Real-time Face Recognition ROS Node
Subscribes to robot camera and publishes face recognition results
"""

import warnings

# InsightFace calls np.linalg.lstsq; NumPy emits FutureWarning per call. If that runs
# every frame, stderr spam can noticeably slow the node.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*rcond.*",
)

import rospy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import cv2
import sys
import os

# Add the face_standalone directory to Python path
sys.path.append('/face_standalone-main')

from face_recognition import FaceRecognitionHandler
from config import FACE_MIN_SIZE_RT


def compressed_imgmsg_to_bgr(msg):
    """Decode CompressedImage without cv_bridge (avoids PyInit_cv_bridge_boost ABI mismatches)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return bgr


def bgr_to_imgmsg(bgr, header=None, encoding="bgr8"):
    """Build sensor_msgs/Image from BGR ndarray without cv_bridge."""
    out = Image()
    if header is not None:
        out.header = header
    out.height, out.width = bgr.shape[:2]
    out.encoding = encoding
    out.is_bigendian = 0
    out.step = bgr.shape[1] * 3
    out.data = bgr.tobytes()
    return out


class RealTimeFaceNode:
    def __init__(self):
        rospy.init_node('real_time_face_recognition', anonymous=True)
        
        # Initialize face recognition
        rospy.loginfo("Initializing face recognition...")
        self.face_handler = FaceRecognitionHandler()
        
        # Use GPU if available, otherwise CPU
        device_id = 0  # Change to -1 for CPU
        if not self.face_handler.load_face_recognition(device_id=device_id):
            rospy.logerr("Failed to load face recognition model!")
            return
        
        rospy.loginfo(f"Face recognition loaded. Known people: {len(self.face_handler.face_db)}")
        
        # Setup subscribers and publishers
        self.image_sub = rospy.Subscriber("/depth_cam/rgb/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1)
        self.results_pub = rospy.Publisher("/face_recognition/results", String, queue_size=1)
        self.debug_pub = rospy.Publisher("/face_recognition/debug", Image, queue_size=1)
        
        # Performance tracking
        self.frame_count = 0
        self.last_time = rospy.Time.now()
        
        rospy.loginfo("Real-time face recognition node ready!")
        rospy.loginfo("Subscribed to: /depth_cam/rgb/image_raw/compressed")
        rospy.loginfo("Publishing results to: /face_recognition/results")
    
    def image_callback(self, msg):
        try:
            # Convert compressed ROS Image to OpenCV (no cv_bridge)
            cv_image = compressed_imgmsg_to_bgr(msg)
            if cv_image is None:
                rospy.logwarn_throttle(5.0, "compressed image decode failed (cv2.imdecode returned None)")
                return
            
            # Process frame for faces
            recognized_names = self.recognize_faces(cv_image)
            
            # Publish results
            result_str = f"{msg.header.stamp.to_sec()}: {', '.join(recognized_names) if recognized_names else 'No faces'}"
            self.results_pub.publish(result_str)
            
            # Performance tracking
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Every 30 frames
                current_time = rospy.Time.now()
                fps = 30.0 / (current_time - self.last_time).to_sec()
                rospy.loginfo(f"Processing FPS: {fps:.1f}")
                self.last_time = current_time
            
            # Publish debug image with bounding boxes (optional)
            debug_image = self.draw_face_boxes(cv_image, recognized_names)
            debug_msg = bgr_to_imgmsg(debug_image, header=msg.header)
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def recognize_faces(self, frame):
        """Recognize faces in frame using existing face_recognition.py logic"""
        try:
            faces = self.face_handler.face_app.get(frame)
            
            if len(faces) == 0:
                return []
            
            recognized_names = []
            
            for idx, face in enumerate(faces, 1):
                face_bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = face_bbox
                
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Skip small faces
                if face_width < FACE_MIN_SIZE_RT[0] or face_height < FACE_MIN_SIZE_RT[1]:
                    continue
                
                try:
                    embedding = face.embedding.flatten() if hasattr(face.embedding, 'flatten') else face.embedding
                    name = self.face_handler.identify_face(embedding)
                except Exception as e:
                    rospy.logdebug(f"Error identifying face: {e}")
                    name = None
                
                if name:
                    recognized_names.append(name)
                else:
                    recognized_names.append("Unknown")
            
            return recognized_names
            
        except Exception as e:
            rospy.logerr(f"Error in face recognition: {e}")
            return []
    
    def draw_face_boxes(self, frame, names):
        """Draw bounding boxes and names on frame"""
        try:
            faces = self.face_handler.face_app.get(frame)
            result_frame = frame.copy()
            
            for i, face in enumerate(faces):
                if i < len(names):
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Draw rectangle
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw name
                    name = names[i]
                    cv2.putText(result_frame, name, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return result_frame
        except:
            return frame

if __name__ == '__main__':
    try:
        node = RealTimeFaceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Face recognition node stopped")
