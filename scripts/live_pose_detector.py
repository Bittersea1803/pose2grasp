import os
import sys
import time
import argparse
from typing import Optional, List, Tuple
from collections import Counter

import cv2
import numpy as np
import rospy
import torch
import message_filters
import joblib
import pandas as pd

from std_msgs.msg import String as StringMsg
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# --- Path Configuration ---
def get_project_root():
    script_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))

PROJECT_ROOT = get_project_root()
OPENPOSE_PYTHON_PATH = os.path.join(PROJECT_ROOT, "src", "pytorch-openpose")
MODELS_DIR_ROOT = os.path.join(PROJECT_ROOT, "models")

# --- OpenPose Import ---
try:
    sys.path.append(OPENPOSE_PYTHON_PATH)
    from src.hand import Hand
    rospy.loginfo("Successfully imported OpenPose modules.")
except ImportError as e:
    rospy.logfatal(f"Cannot import OpenPose. Details: {e}")
    sys.exit(1)

# --- Configuration Constants ---
RGB_TOPIC = "/camera/rgb/image_rect_color"
REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"
PREDICTED_GRASP_TOPIC = "/predicted_grasp"

# --- LOGIC PARAMETERS FOR VOTING ---
CONFIDENCE_THRESHOLD = 0.75         
VOTE_COLLECTION_DURATION_S = 1.5    
MIN_VOTES_FOR_DECISION = 5          
COOLDOWN_PERIOD_S = 5.0             

OPENPOSE_MODEL_FILE = os.path.join(OPENPOSE_PYTHON_PATH, "model", "hand_pose_model.pth")
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
VALID_DEPTH_THRESHOLD_MM = (400, 1500)


class LiveGraspDetector:
    def __init__(self, model_type="xgboost"):
        self.bridge = CvBridge()
        self._camera_info_received = False
        self._last_processed_time = time.time()
        self.model_type = model_type.lower()
        self.feature_names = [f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')]

        # --- State Machine and Voting Variables ---
        self.state = "WAITING"  # Stanja: WAITING, COLLECTING
        self.votes = []
        self.collection_start_time = 0
        self.cooldown_until = 0
        rospy.loginfo(f"Detector initialized. State: {self.state}")
        # --- End of State Machine Variables ---

        self._load_models()
        self._initialize_openpose()

        self.grasp_pub = rospy.Publisher(PREDICTED_GRASP_TOPIC, StringMsg, queue_size=10)
        rospy.loginfo(f"Publishing final grasp decision on topic: {PREDICTED_GRASP_TOPIC}")

        self.rgb_sub = message_filters.Subscriber(RGB_TOPIC, Image)
        self.depth_sub = message_filters.Subscriber(REGISTERED_DEPTH_TOPIC, Image)
        self.info_sub = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self._synchronized_callback)
        rospy.loginfo("Live Grasp Detector node running. Waiting for synchronized data...")

    def _load_models(self):
        model_subdir = os.path.join(MODELS_DIR_ROOT, self.model_type)
        model_path = os.path.join(model_subdir, f"{self.model_type}_model.joblib" if self.model_type == "xgboost" else f"best_{self.model_type}_model.joblib")
        encoder_path = os.path.join(model_subdir, f"label_encoder_{self.model_type}.joblib")
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            if self.model_type == "random_forest":
                imputer_path = os.path.join(model_subdir, f"imputer_{self.model_type}.joblib")
                self.imputer = joblib.load(imputer_path)
        except FileNotFoundError as e:
            rospy.logfatal(f"Error loading models: {e}.")
            sys.exit(1)


    def _initialize_openpose(self):
        try:
            self.hand_estimator = Hand(OPENPOSE_MODEL_FILE)
        except Exception as e:
            rospy.logfatal(f"Failed to initialize OpenPose: {e}")
            sys.exit(1)


    def _synchronized_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        current_time = time.time()
        if self.state == "WAITING" and current_time < self.cooldown_until:
            status_msg = f"\rCOOLDOWN active. Waiting for {self.cooldown_until - current_time:.1f}s... "
            sys.stdout.write(status_msg)
            sys.stdout.flush()
            return
        # ---

        if not self._camera_info_received:
            self._fx, self._fy = info_msg.K[0], info_msg.K[4]
            self._cx, self._cy = info_msg.K[2], info_msg.K[5]
            self._camera_info_received = True
            rospy.loginfo_once(f"Camera intrinsics received.")

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            self._process_frame_and_vote(cv_rgb, cv_depth)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def _get_3d_keypoints(self, rgb_frame, depth_frame):
        all_peaks_2d = self.hand_estimator(rgb_frame)
        if not isinstance(all_peaks_2d, np.ndarray) or all_peaks_2d.shape[0] == 0: return None, 0
        
        peaks_2d_conf = np.full((21, 2), np.nan, dtype=np.float32)
        for i in range(min(21, all_peaks_2d.shape[0])):
            if all_peaks_2d[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD:
                peaks_2d_conf[i, :] = all_peaks_2d[i, :2]
        
        if np.sum(~np.isnan(peaks_2d_conf)) < 13 * 2: return None, 0

        keypoints_3d_cam = np.full((21, 3), np.nan, dtype=np.float32)
        for i, (x_px, y_px) in enumerate(peaks_2d_conf):
            if not np.isnan(x_px):
                h, w = depth_frame.shape
                if 0 <= int(y_px) < h and 0 <= int(x_px) < w:
                    z_mm = depth_frame[int(y_px), int(x_px)]
                    if VALID_DEPTH_THRESHOLD_MM[0] < z_mm < VALID_DEPTH_THRESHOLD_MM[1]:
                        z_m = z_mm / 1000.0
                        keypoints_3d_cam[i] = [(x_px - self._cx) * z_m / self._fx, (y_px - self._cy) * z_m / self._fy, z_m]
        
        if np.isnan(keypoints_3d_cam[0]).any(): return None, 0
        keypoints_3d_rel = keypoints_3d_cam - keypoints_3d_cam[0]
        num_valid = np.sum(~np.isnan(keypoints_3d_rel).any(axis=1))
        return keypoints_3d_rel, num_valid

    def _process_frame_and_vote(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        keypoints_3d, num_valid_points = self._get_3d_keypoints(rgb_frame, depth_frame)

        if keypoints_3d is None:
            if self.state == "COLLECTING":
                if time.time() - self.collection_start_time > VOTE_COLLECTION_DURATION_S:
                    self._determine_winner() # Završi glasanje
            else:
                sys.stdout.write(f"\r{self.state}: No hand detected. ")
                sys.stdout.flush()
            return

        feature_vector = keypoints_3d.flatten()
        features_df = pd.DataFrame([feature_vector], columns=self.feature_names)
        if self.model_type == "random_forest":
            features_df = pd.DataFrame(self.imputer.transform(features_df), columns=self.feature_names)

        probabilities = self.model.predict_proba(features_df)[0]
        max_prob = np.max(probabilities)
        
        current_time = time.time()

        if self.state == "WAITING":
            if max_prob >= CONFIDENCE_THRESHOLD:
                self.state = "COLLECTING"
                self.collection_start_time = current_time
                self.votes = []
                prediction_index = np.argmax(probabilities)
                first_vote = self.label_encoder.inverse_transform([prediction_index])[0]
                self.votes.append(first_vote)
                rospy.loginfo(f"Hand detected with confidence {max_prob:.2f}. Starting vote collection for {VOTE_COLLECTION_DURATION_S}s.")

        elif self.state == "COLLECTING":
            if max_prob >= CONFIDENCE_THRESHOLD:
                prediction_index = np.argmax(probabilities)
                vote = self.label_encoder.inverse_transform([prediction_index])[0]
                self.votes.append(vote)

            time_left = VOTE_COLLECTION_DURATION_S - (current_time - self.collection_start_time)
            status_msg = f"\rCOLLECTING... Votes: {len(self.votes)}. Time left: {max(0, time_left):.1f}s"
            sys.stdout.write(status_msg)
            sys.stdout.flush()

            if current_time - self.collection_start_time > VOTE_COLLECTION_DURATION_S:
                self._determine_winner()
                
    def _determine_winner(self):
        rospy.loginfo("Vote collection finished. Determining winner...")
        
        if len(self.votes) >= MIN_VOTES_FOR_DECISION:
            # Prebroji glasove
            vote_counts = Counter(self.votes)
            # Nađi pobjednika
            winner, count = vote_counts.most_common(1)[0]
            
            rospy.loginfo(f"Final Decision: '{winner.upper()}' with {count} votes. Publishing to topic.")
            print(f"\nFINAL GRASP: {winner.upper()}")
            self.grasp_pub.publish(winner)
            
        else:
            rospy.logwarn(f"Failed to reach decision. Needed {MIN_VOTES_FOR_DECISION} votes, got {len(self.votes)}.")
            print("\nDECISION FAILED: Not enough confident detections.")

        self.state = "WAITING"
        self.votes = []
        self.cooldown_until = time.time() + COOLDOWN_PERIOD_S
        rospy.loginfo(f"Returning to WAITING state. Cooldown for {COOLDOWN_PERIOD_S}s.")


    def run(self):
        rospy.spin()
        print("\nLive Grasp Detector node shutting down.")


if __name__ == "__main__":
    rospy.init_node("live_grasp_detector_node", anonymous=True)
    parser = argparse.ArgumentParser(description="Live grasp detection with voting logic.")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "random_forest"])
    args = parser.parse_args(rospy.myargv()[1:])

    try:
        detector = LiveGraspDetector(model_type=args.model)
        detector.run()
    except rospy.ROSInterruptException:
        pass