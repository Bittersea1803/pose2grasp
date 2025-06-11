#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import rospy
import torch
import joblib
import pandas as pd
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from collections import deque, Counter
from typing import Optional, List, Tuple

# --- Automatsko postavljanje putanja ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_SRC_DIR = os.path.dirname(SCRIPT_DIR) 
    PACKAGE_ROOT_DIR = os.path.dirname(PACKAGE_SRC_DIR)
    OPENPOSE_PATH = os.path.join(PACKAGE_ROOT_DIR, 'pytorch-openpose')
    sys.path.append(OPENPOSE_PATH)
    from src.hand import Hand
    rospy.loginfo(f"Successfully imported OpenPose from: {OPENPOSE_PATH}")
except ImportError:
    rospy.logfatal(f"Could not find 'pytorch-openpose' directory. Please check the folder structure.")
    sys.exit(1)


def nothing(x):
    pass

# --- Konstante ---
GRASP_DECISION_TOPIC = "/pose2grasp/grasp_type"
HAND_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
VALID_DEPTH_THRESHOLD_MM = (400, 1500)
DEPTH_NEIGHBORHOOD_SIZE = 3
MIN_VALID_KEYPOINTS = 13
VOTING_WINDOW_SIZE = 10
VOTING_AGREEMENT_PERCENTAGE = 0.7

class LiveGraspDetector:
    def __init__(self, model_type="xgboost"):
        rospy.init_node("live_grasp_detector_node", anonymous=True)
        self.bridge = CvBridge()
        self.is_shutdown = False
        
        # --- Konfiguracija kamere ---
        self.RGB_TOPIC = "/camera/rgb/image_rect_color"
        self.DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
        self.INFO_TOPIC = "/camera/rgb/camera_info"
        
        # --- Učitavanje modela s relativnim putanjama ---
        self.model_type = model_type.lower()
        models_dir = os.path.join(SCRIPT_DIR, 'models', self.model_type)
        model_path = os.path.join(models_dir, f'{self.model_type}_model.joblib')
        encoder_path = os.path.join(models_dir, f'label_encoder_{self.model_type}.joblib')
        
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            rospy.loginfo(f"Successfully loaded model from: {model_path}")
        except FileNotFoundError as e:
            rospy.logfatal(f"FATAL: Could not load model files. Error: {e}")
            sys.exit(1)
        
        self.feature_names = [f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')]

        # --- Inicijalizacija OpenPose ---
        openpose_model_file = os.path.join(OPENPOSE_PATH, "model", "hand_pose_model.pth")
        self._hand_estimator = Hand(openpose_model_file)
        self._openpose_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._hand_estimator.model = self._hand_estimator.model.to(self._openpose_device)
        rospy.loginfo(f"OpenPose running on device: {self._openpose_device}")

        # --- ROS ---
        self.grasp_decision_publisher = rospy.Publisher(GRASP_DECISION_TOPIC, String, queue_size=10)
        rgb_sub = message_filters.Subscriber(self.RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(self.DEPTH_TOPIC, Image)
        info_sub = message_filters.Subscriber(self.INFO_TOPIC, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._callback)

        # --- GUI i Varijable Stanja ---
        self.cv_window_name = "Live Grasp Detector"
        self._init_gui()
        self._init_state_variables()

        rospy.loginfo("Live Grasp Detector node running. Waiting for data...")

    def _init_gui(self):
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('OpenPose Conf', self.cv_window_name, 20, 100, nothing) 
        cv2.createTrackbar('Voting Conf', self.cv_window_name, 60, 100, nothing)

    def _init_state_variables(self):
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.camera_info_received = False
        self._vote_buffer = deque(maxlen=VOTING_WINDOW_SIZE)
        self._current_stable_prediction: Optional[str] = None
        self._min_votes_for_decision = int(VOTING_WINDOW_SIZE * VOTING_AGREEMENT_PERCENTAGE)

    def _draw_gui(self, frame, status, raw_label, confidence, voted_status, keypoints_2d):
        # Iscrtaj kostur ruke
        if keypoints_2d is not None:
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if not np.isnan(keypoints_2d[p1_idx]).any() and not np.isnan(keypoints_2d[p2_idx]).any():
                    p1 = tuple(keypoints_2d[p1_idx].astype(int)); p2 = tuple(keypoints_2d[p2_idx].astype(int))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
            for i in range(keypoints_2d.shape[0]):
                if not np.isnan(keypoints_2d[i]).any():
                    pt = tuple(keypoints_2d[i].astype(int)); cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        
        # Ispiši status
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if "ERROR" in status else (255, 255, 255), 2)
        cv2.putText(frame, f"Raw: {raw_label.upper()} ({confidence:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Voted: {voted_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(self.cv_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.is_shutdown = True
            rospy.signal_shutdown("User pressed 'q' to exit.")

    def _get_depth_for_point(self, depth_map_mm, point_2d):
        u, v = int(round(point_2d[0])), int(round(point_2d[1]))
        if not (0 <= v < depth_map_mm.shape[0] and 0 <= u < depth_map_mm.shape[1]): return np.nan
        
        depth_val = depth_map_mm[v, u]
        if VALID_DEPTH_THRESHOLD_MM[0] < depth_val < VALID_DEPTH_THRESHOLD_MM[1]:
            return float(depth_val) / 1000.0
        return np.nan

    def _callback(self, rgb_msg, depth_msg, info_msg):
        if not self.camera_info_received:
            K=info_msg.K; self.fx,self.fy,self.cx,self.cy=K[0],K[4],K[2],K[5]; self.camera_info_received=True
            rospy.loginfo_once(f"Camera intrinsics received")

        display_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        status, raw_label, confidence, voted_status, keypoints_2d = "OK", "N/A", 0.0, "Waiting...", None

        try:
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            openpose_conf = cv2.getTrackbarPos('OpenPose Conf', self.cv_window_name) / 100.0
            
            # 1. Detekcija i filtriranje 2D točaka
            all_peaks = self._hand_estimator(rgb_frame)
            if not isinstance(all_peaks, np.ndarray) or all_peaks.shape[0]==0: raise ValueError("No hand detected")
            
            keypoints_2d = np.full((21, 2), np.nan)
            for i in range(min(21, all_peaks.shape[0])):
                if all_peaks[i, 2] >= openpose_conf: keypoints_2d[i] = all_peaks[i, :2]

            # 2. Projekcija u 3D
            z = np.array([self._get_depth_for_point(depth_frame, p) for p in keypoints_2d])
            x = (keypoints_2d[:, 0] - self.cx) * z / self.fx
            y = (keypoints_2d[:, 1] - self.cy) * z / self.fy
            keypoints_3d = np.stack((x, y, z), axis=-1)

            # 3. Konverzija u relativne koordinate i predikcija
            if np.isnan(keypoints_3d[0]).any(): raise ValueError("Wrist not in 3D")
            keypoints_rel = keypoints_3d - keypoints_3d[0]
            if np.sum(~np.isnan(keypoints_rel)) < MIN_VALID_KEYPOINTS*3: raise ValueError("Not enough 3D points")

            features = pd.DataFrame([keypoints_rel.flatten()], columns=self.feature_names)
            pred_num = self.model.predict(features)
            raw_label = self.label_encoder.inverse_transform(pred_num)[0]
            probs = self.model.predict_proba(features)[0]
            confidence = probs[pred_num[0]]

        except Exception as e:
            status = f"ERROR: {e}"

        # 4. Logika glasanja
        voting_conf = cv2.getTrackbarPos('Voting Conf', self.cv_window_name) / 100.0
        if status == "OK" and confidence >= voting_conf: self._vote_buffer.append(raw_label)
        
        if len(self._vote_buffer) == self._vote_buffer.maxlen:
            counts = Counter(self._vote_buffer); most_common, num = counts.most_common(1)[0]
            if num >= self._min_votes_for_decision:
                voted_status = f"Stable: {most_common}"
                if self._current_stable_prediction != most_common:
                    self._current_stable_prediction = most_common
                    self.grasp_decision_publisher.publish(String(data=most_common))
            else:
                voted_status = "Unstable"; self._current_stable_prediction = None
        else:
            voted_status = f"Collecting ({len(self._vote_buffer)}/{VOTING_WINDOW_SIZE})"

        # 5. Crtanje GUI-a
        self._draw_gui(display_frame, status, raw_label, confidence, voted_status, keypoints_2d)

if __name__ == "__main__":
    try:
        detector = LiveGraspDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nLive Grasp Detector shutting down.")