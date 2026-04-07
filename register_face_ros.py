#!/usr/bin/env python3
"""
Subscribe to a ROS camera topic, sample frames at 1 Hz, and append face embeddings
to the static database (same JSON as register_face.py). Re-run with the same name
to add more embeddings and improve matching.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*rcond.*",
)

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from register_face import FaceRegistration


def compressed_imgmsg_to_bgr(msg):
    """Decode CompressedImage without cv_bridge."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def raw_imgmsg_to_bgr(msg):
    """Decode sensor_msgs/Image (bgr8 or rgb8) without cv_bridge."""
    enc = (msg.encoding or "").lower()
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "bgr8":
        return arr.reshape((msg.height, msg.width, 3))
    if enc == "rgb8":
        rgb = arr.reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rospy.logerr_throttle(
        5.0,
        f"Unsupported image encoding '{msg.encoding}' (use bgr8 or rgb8, or compressed JPEG)",
    )
    return None


class RosFaceCapture:
    def __init__(self, person_name, duration_sec, topic, compressed):
        rospy.init_node("register_face_ros", anonymous=True)
        self.person_name = person_name
        self.reg = FaceRegistration()
        self.end_time = rospy.Time.now() + rospy.Duration(duration_sec)
        self.last_capture_time = rospy.Time(0)
        self.capture_ok = 0
        self._finished = False
        self._logged_waiting = False

        rospy.on_shutdown(self._on_shutdown)

        if compressed:
            self._sub = rospy.Subscriber(
                topic, CompressedImage, self._cb_compressed, queue_size=1
            )
            rospy.loginfo("Subscribing (CompressedImage): %s", topic)
        else:
            self._sub = rospy.Subscriber(topic, Image, self._cb_raw, queue_size=1)
            rospy.loginfo("Subscribing (Image): %s", topic)

        # Exit when duration elapses even if the camera stops publishing.
        self._deadline_timer = rospy.Timer(rospy.Duration(0.25), self._check_deadline)

        rospy.loginfo(
            "Capturing for %.1f s at 1 Hz; stand in view and vary angle slightly.",
            duration_sec,
        )

    def _on_shutdown(self):
        self._finish_once()

    def _check_deadline(self, _evt=None):
        if self._finished or rospy.is_shutdown():
            return
        if rospy.Time.now() <= self.end_time:
            return
        self._finish_once()
        rospy.signal_shutdown("capture duration elapsed")

    def _finish_once(self):
        if self._finished:
            return
        self._finished = True
        self.reg.face_handler.reload_face_database()
        print(
            f"Session finished: {self.capture_ok} face(s) captured successfully for '{self.person_name}'."
        )

    def _maybe_capture(self, bgr):
        if bgr is None:
            return
        now = rospy.Time.now()
        if now > self.end_time:
            return

        if not self._logged_waiting:
            self._logged_waiting = True
            print("Receiving camera frames; sampling at 1 Hz…")

        dt = (now - self.last_capture_time).to_sec()
        if dt < 1.0 - 1e-3:
            return
        self.last_capture_time = now

        ok, msg = self.reg.try_register_frame(bgr, self.person_name)
        if ok:
            self.capture_ok += 1
            print("face captured successfully")
        else:
            print(f"skip ({msg})")

    def _cb_compressed(self, ros_msg):
        if rospy.is_shutdown():
            return
        bgr = compressed_imgmsg_to_bgr(ros_msg)
        if bgr is None:
            rospy.logwarn_throttle(5.0, "compressed image decode failed")
            return
        self._maybe_capture(bgr)

    def _cb_raw(self, ros_msg):
        if rospy.is_shutdown():
            return
        bgr = raw_imgmsg_to_bgr(ros_msg)
        self._maybe_capture(bgr)


def main():
    parser = argparse.ArgumentParser(
        description="Append face embeddings from ROS camera at 1 Hz for a given duration."
    )
    parser.add_argument(
        "name",
        help="Person name (stored in static face DB; repeat runs append embeddings)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        metavar="SEC",
        help="How long to run and capture (default: 30)",
    )
    parser.add_argument(
        "--topic",
        default="/depth_cam/rgb/image_raw/compressed",
        help="Image topic (default: depth camera compressed)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Subscribe to sensor_msgs/Image (bgr8/rgb8) instead of CompressedImage",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        print("Error: --duration must be positive")
        sys.exit(1)

    RosFaceCapture(args.name, args.duration, args.topic, compressed=not args.raw)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
