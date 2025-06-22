#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import torch
import joblib
from collections import deque, Counter
import os
import sys

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from core.real_ur5_controller import UR5Controller

CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"
OUTPUT_FILENAME = "pose_result.txt"

HOME_POSE_JOINTS = np.deg2rad([-89, -6, -140, -54, 91, 45])

# -- Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..') 
OPENPOSE_PATH = os.path.join(PACKAGE_ROOT_DIR, 'src', 'pytorch-openpose')
sys.path.append(OPENPOSE_PATH)
from src.hand import Hand
rospy.loginfo("Successfully configured paths and imported OpenPose.")

VALID_DEPTH_THRESHOLD_MM = (400, 1500)
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
MIN_VALID_KEYPOINTS_FOR_PREDICTION = 15
OUTLIER_XYZ_THRESHOLD_M = 0.25
MAX_LIMB_LENGTH_M = 0.10
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
MEDIAN_FILTER_KERNEL_SIZE = 3

VOTING_WINDOW_SIZE = 15
VOTING_AGREEMENT_PERCENTAGE = 0.7

# -- Hand Skeleton
HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
                    [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

class HandPoseClassifierLogic:
    def __init__(self, model_path, encoder_path, hand_model_path):
        self.hand_estimator = Hand(hand_model_path)
        self.classifier_model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.fx = self.fy = self.cx = self.cy = None
        rospy.loginfo("HandPoseClassifierLogic initialized with ML models.")

    def set_camera_intrinsics(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        rospy.loginfo(f"Camera intrinsics set: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    def process_frame(self, rgb_image, depth_image):
        if self.fx is None: raise ValueError("Camera intrinsics not set.")
        depth_filtered = cv2.medianBlur(depth_image, MEDIAN_FILTER_KERNEL_SIZE)
        all_peaks = self.hand_estimator(rgb_image)
        if all_peaks is None: return None
        
        has_confidence = all_peaks.shape[1] == 3
        keypoints_2d = np.full((21, 2), np.nan, dtype=np.float32)
        for i in range(min(21, all_peaks.shape[0])):
            if not has_confidence or (has_confidence and all_peaks[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD):
                keypoints_2d[i] = all_peaks[i, :2]
        
        keypoints_3d = self._project_points_to_3d(keypoints_2d, depth_filtered)
        if np.isnan(keypoints_3d[0]).any(): return None
        
        relative_kps, mask = self._filter_3d_outliers(keypoints_3d - keypoints_3d[0])
        relative_kps, mask = self._filter_3d_by_limb_length(relative_kps, mask)
        
        if np.sum(mask) < MIN_VALID_KEYPOINTS_FOR_PREDICTION: return None
        
        features = np.nan_to_num(relative_kps.flatten()).reshape(1, -1)
        try:
            pred_encoded = self.classifier_model.predict(features)
            return self.label_encoder.inverse_transform(pred_encoded)[0]
        except (IndexError, ValueError): return None

    def _project_points_to_3d(self, keypoints_2d, depth_map):
        keypoints_3d = np.full((21, 3), np.nan, dtype=np.float32)
        for i, (u, v) in enumerate(keypoints_2d):
            if not np.isnan(u):
                z = self.get_depth(depth_map, int(u), int(v))
                if not np.isnan(z):
                    x, y = (u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy
                    keypoints_3d[i] = [x, y, z]
        return keypoints_3d

    def get_depth(self, depth_map_mm, u_px, v_px):
        if not (0 <= v_px < depth_map_mm.shape[0] and 0 <= u_px < depth_map_mm.shape[1]): return np.nan
        radius = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_s, y_e = max(0, v_px - radius), min(depth_map_mm.shape[0], v_px + radius + 1)
        x_s, x_e = max(0, u_px - radius), min(depth_map_mm.shape[1], u_px + radius + 1)
        neighborhood = depth_map_mm[y_s:y_e, x_s:x_e]
        valid_depths = neighborhood[(neighborhood >= VALID_DEPTH_THRESHOLD_MM[0]) & (neighborhood <= VALID_DEPTH_THRESHOLD_MM[1])]
        if valid_depths.size < max(1, (DEPTH_NEIGHBORHOOD_SIZE**2) // 4): return np.nan
        if np.std(valid_depths) > DEPTH_STD_DEV_THRESHOLD_MM: return np.nan
        return float(np.median(valid_depths)) / 1000.0

    def _filter_3d_outliers(self, keypoints_3d_relative):
        filtered_points = keypoints_3d_relative.copy()
        validity_mask = ~np.isnan(filtered_points).any(axis=1)
        if not validity_mask[0]: return filtered_points, validity_mask.tolist()
        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21):
            if validity_mask[i] and np.sum(filtered_points[i]**2) > max_dist_sq:
                filtered_points[i, :] = np.nan
                validity_mask[i] = False
        return filtered_points, validity_mask.tolist()

    def _filter_3d_by_limb_length(self, keypoints_3d_rel, validity_mask):
        points, mask = keypoints_3d_rel.copy(), list(validity_mask)
        for _ in range(3):
            removed = 0
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if mask[p1_idx] and mask[p2_idx]:
                    if np.sum((points[p1_idx] - points[p2_idx])**2) > MAX_LIMB_LENGTH_M**2:
                        d1_sq, d2_sq = np.sum(points[p1_idx]**2), np.sum(points[p2_idx]**2)
                        remove_idx = p1_idx if d1_sq > d2_sq else p2_idx
                        if mask[remove_idx]:
                            points[remove_idx, :] = np.nan
                            mask[remove_idx] = False
                            removed += 1
            if removed == 0: break
        return points, mask

class PoseDetectorNode:
    def __init__(self):
        rospy.init_node('pose_detector_node', anonymous=True)
        rospy.loginfo("Initializing Pose Detector Node...")

        model_type = 'xgboost'
        rospy.loginfo(f"Using classifier model type: {model_type}")
        
        src_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(src_dir, 'models', model_type)
        model_path = os.path.join(models_dir, f'{model_type}_model.joblib')
        encoder_path = os.path.join(models_dir, f'label_encoder_{model_type}.joblib')
        hand_model_path = os.path.join(OPENPOSE_PATH, "model", "hand_pose_model.pth")

        self.classifier_logic = HandPoseClassifierLogic(model_path, encoder_path, hand_model_path)
        self.robot_controller = UR5Controller()
        
        self.bridge = CvBridge()
        self.vote_buffer = deque(maxlen=VOTING_WINDOW_SIZE)
        self.min_votes = int(VOTING_WINDOW_SIZE * VOTING_AGREEMENT_PERCENTAGE)
        self.task_complete = False
        
        self.image_subscriber = None

    def run(self):
        # 1. Move robot to HOME position
        rospy.loginfo("Step 1: Moving robot to HOME_POSE for pose detection.")
        joint_trajectory_points = np.array([
            self.robot_controller.get_current_joint_values(), 
            HOME_POSE_JOINTS
        ])
        self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)
        rospy.loginfo("Robot is at HOME_POSE.")
        
        # 2. Initialize camera and start listening
        rospy.loginfo("Step 2: Initializing camera listeners.")
        try:
            cam_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
            self.classifier_logic.set_camera_intrinsics(cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5])
        except rospy.ROSException:
            rospy.logfatal("Timeout waiting for camera_info. Is the camera running?")
            sys.exit(1)
        
        rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
        self.image_subscriber = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.2)
        self.image_subscriber.registerCallback(self.synchronized_callback)
        
        rospy.loginfo("--- Pose Detector is now active. Waiting for a stable hand pose... ---")
        rospy.spin() # Keep the node alive until shutdown is called

    def synchronized_callback(self, rgb_msg, depth_msg):
        if self.task_complete:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Bridge conversion error: {e}")
            return
            
        label = self.classifier_logic.process_frame(rgb_image, depth_image)
        
        if label is None:
            if len(self.vote_buffer) > 0:
                rospy.loginfo("Voting: Hand lost or invalid, clearing vote buffer.")
                self.vote_buffer.clear()
        else:
            self.vote_buffer.append(label)
            rospy.loginfo(f"Voting: Added '{label}' to buffer. Buffer size: {len(self.vote_buffer)}/{VOTING_WINDOW_SIZE}")

        if len(self.vote_buffer) >= VOTING_WINDOW_SIZE:
            counts = Counter(self.vote_buffer)
            rospy.loginfo(f"Voting: Buffer full. Counts: {counts}")
            most_common, count = counts.most_common(1)[0]
            
            if count >= self.min_votes:
                self.write_result_and_shutdown(most_common)

    def write_result_and_shutdown(self, label):
        if self.task_complete: return
        self.task_complete = True
        rospy.loginfo(f"--- STABLE POSE DETECTED: '{label}' ---")
        try:
            output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILENAME)
            with open(output_path, 'w') as f:
                f.write(label)
            rospy.loginfo(f"Successfully wrote '{label}' to '{output_path}'.")
        except IOError as e:
            rospy.logfatal(f"Failed to write to file {output_path}: {e}")
        
        rospy.loginfo("Shutting down node.")
        rospy.signal_shutdown("Stable pose detected and saved to file.")

if __name__ == '__main__':
    try:
        node = PoseDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("PoseDetectorNode shutdown complete.")
