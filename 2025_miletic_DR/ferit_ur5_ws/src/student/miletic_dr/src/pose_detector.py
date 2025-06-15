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
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters

# --- Configurable Camera Topics ---
CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

# --- Path Configuration ---
try:
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_ROOT_DIR = os.path.join(SRC_DIR, '..')
    OPENPOSE_PATH = os.path.join(PACKAGE_ROOT_DIR, 'pytorch-openpose')
    sys.path.append(OPENPOSE_PATH)
    from src.hand import Hand
except ImportError:
    rospy.logfatal("Could not import OpenPose. Check path in pose_detector.py.")
    sys.exit(1)

# --- Constants ---
VALID_DEPTH_THRESHOLD_MM = (400, 1500)
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
MIN_VALID_KEYPOINTS_FOR_PREDICTION = 15
OUTLIER_XYZ_THRESHOLD_M = 0.25
MAX_LIMB_LENGTH_M = 0.10
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
MEDIAN_FILTER_KERNEL_SIZE = 3

HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
]


class HandPoseClassifierLogic:
    def __init__(self, model_path, encoder_path, hand_model_path):
        self.hand_estimator = Hand(hand_model_path)
        self.classifier_model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.fx = self.fy = self.cx = self.cy = None

    def set_camera_intrinsics(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def process_frame(self, rgb_image, depth_image):
        if self.fx is None:
            raise ValueError("Camera intrinsics are not set.")

        depth_filtered = cv2.medianBlur(depth_image, MEDIAN_FILTER_KERNEL_SIZE)
        all_peaks = self.hand_estimator(rgb_image)
        if all_peaks is None:
            return None

        has_confidence = all_peaks.shape[1] == 3
        keypoints_2d = np.full((21, 2), np.nan, dtype=np.float32)
        for i in range(min(21, all_peaks.shape[0])):
            if (has_confidence and all_peaks[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD) or not has_confidence:
                keypoints_2d[i] = all_peaks[i, :2]

        keypoints_3d = self._project_points_to_3d(keypoints_2d, depth_filtered)
        if np.isnan(keypoints_3d[0]).any():
            return None

        relative_kps, mask = self._filter_3d_outliers(keypoints_3d - keypoints_3d[0])
        relative_kps, mask = self._filter_3d_by_limb_length(relative_kps, mask)

        if np.sum(mask) < MIN_VALID_KEYPOINTS_FOR_PREDICTION:
            return None

        features = np.nan_to_num(relative_kps.flatten()).reshape(1, -1)
        try:
            pred_encoded = self.classifier_model.predict(features)
            return self.label_encoder.inverse_transform(pred_encoded)[0]
        except (IndexError, ValueError):
            return None

    def _project_points_to_3d(self, keypoints_2d, depth_map):
        keypoints_3d = np.full((21, 3), np.nan, dtype=np.float32)
        for i, (u, v) in enumerate(keypoints_2d):
            if not np.isnan(u):
                z = self.get_robust_depth(depth_map, int(u), int(v))
                if not np.isnan(z):
                    x, y = (u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy
                    keypoints_3d[i] = [x, y, z]
        return keypoints_3d

    def get_robust_depth(self, depth_map_mm, u_px, v_px):
        if not (0 <= v_px < depth_map_mm.shape[0] and 0 <= u_px < depth_map_mm.shape[1]):
            return np.nan
        radius = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_start, y_end = max(0, v_px-radius), min(depth_map_mm.shape[0], v_px+radius+1)
        x_start, x_end = max(0, u_px-radius), min(depth_map_mm.shape[1], u_px+radius+1)
        neighborhood = depth_map_mm[y_start:y_end, x_start:x_end]
        valid_depths = neighborhood[(neighborhood >= VALID_DEPTH_THRESHOLD_MM[0]) & (neighborhood <= VALID_DEPTH_THRESHOLD_MM[1])]
        if valid_depths.size < max(1, (DEPTH_NEIGHBORHOOD_SIZE**2) // 4):
            return np.nan
        if np.std(valid_depths) > DEPTH_STD_DEV_THRESHOLD_MM:
            return np.nan
        return float(np.median(valid_depths)) / 1000.0

    def _filter_3d_outliers(self, keypoints_3d_relative):
        filtered_points = keypoints_3d_relative.copy()
        validity_mask = ~np.isnan(filtered_points).any(axis=1)
        if not validity_mask[0]:
            return filtered_points, validity_mask.tolist()
        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21):
            if validity_mask[i] and np.sum(filtered_points[i]**2) > max_dist_sq:
                filtered_points[i], validity_mask[i] = np.nan, False
        return filtered_points, validity_mask.tolist()

    def _filter_3d_by_limb_length(self, keypoints_3d_rel, validity_mask):
        points, mask = keypoints_3d_rel.copy(), list(validity_mask)
        for _ in range(3):
            removed = 0
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if mask[p1_idx] and mask[p2_idx]:
                    if np.sum((points[p1_idx] - points[p2_idx])**2) > MAX_LIMB_LENGTH_M**2:
                        d1, d2 = np.sum(points[p1_idx]**2), np.sum(points[p2_idx]**2)
                        remove_idx = p1_idx if d1 > d2 else p2_idx
                        if mask[remove_idx]:
                            points[remove_idx], mask[remove_idx] = np.nan, False
                            removed += 1
            if removed == 0:
                break
        return points, mask


class PoseDetectorNode:
    def __init__(self):
        rospy.init_node('pose_detector_node')

        model_type = rospy.get_param('~model_type', 'xgboost')

        models_dir = os.path.join(SRC_DIR, 'models', model_type)
        model_path = os.path.join(models_dir, f'{model_type}_model.joblib')
        encoder_path = os.path.join(models_dir, f'label_encoder_{model_type}.joblib')
        hand_model_path = os.path.join(OPENPOSE_PATH, "model", "hand_pose_model.pth")

        self.classifier_logic = HandPoseClassifierLogic(model_path, encoder_path, hand_model_path)

        try:
            cam_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
        except rospy.ROSException:
            rospy.logfatal("Timeout waiting for camera_info. Is the camera running?")
            sys.exit(1)

        self.classifier_logic.set_camera_intrinsics(cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5])

        self.bridge = CvBridge()
        self.vote_buffer = deque(maxlen=rospy.get_param('~voter_window', 15))
        self.min_votes = int(self.vote_buffer.maxlen * rospy.get_param('~voter_agreement', 0.8))
        self.last_published_grasp = None

        self.grasp_pub = rospy.Publisher("/pose2grasp/grasp_type", String, queue_size=10)

        rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.synchronized_callback)

        rospy.loginfo("Pose Detector Node initialized and running.")

    def synchronized_callback(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Bridge conversion error: {e}")
            return

        label = self.classifier_logic.process_frame(rgb, depth)
        self.vote_buffer.append(label)

        if len(self.vote_buffer) < self.vote_buffer.maxlen:
            return

        counts = Counter(self.vote_buffer)
        most_common, count = counts.most_common(1)[0]

        if most_common is not None and count >= self.min_votes:
            if most_common != self.last_published_grasp:
                self.grasp_pub.publish(String(data=most_common))
                self.last_published_grasp = most_common
                rospy.loginfo(f"Published new stable grasp: {most_common}")


if __name__ == '__main__':
    try:
        PoseDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass