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

try:
    PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OPENPOSE_PATH = os.path.join(PACKAGE_ROOT_DIR, 'pytorch-openpose')
    sys.path.append(OPENPOSE_PATH)
    from src.hand import Hand
except ImportError as e:
    rospy.logfatal(f"Could not import OpenPose. Provjerite putanju u pose_detector.py. Greška: {e}")
    sys.exit(1)


# CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
# CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
# CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"

GRASP_TYPE_TOPIC = "/miletic_dr/pose_type"

VALID_DEPTH_THRESHOLD_MM = (200, 1000)
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
MIN_VALID_KEYPOINTS_FOR_PREDICTION = 15
OUTLIER_XYZ_THRESHOLD_M = 0.25
MAX_LIMB_LENGTH_M = 0.10
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
MEDIAN_FILTER_KERNEL_SIZE = 3
HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

class HandPoseClassifierLogic:
    """ Sadrži svu logiku za obradu slike i klasifikaciju, odvojeno od ROS-a. """
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
        if all_peaks is None: return None, None

        has_confidence = all_peaks.shape[1] == 3
        keypoints_2d = np.full((21, 2), np.nan, dtype=np.float32)
        for i in range(min(21, all_peaks.shape[0])):
            if not has_confidence or (has_confidence and all_peaks[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD):
                keypoints_2d[i] = all_peaks[i, :2]

        keypoints_3d = self._project_points_to_3d(keypoints_2d, depth_filtered)
        if np.isnan(keypoints_3d[0]).any(): return None, None

        relative_kps, mask = self._filter_3d_outliers(keypoints_3d - keypoints_3d[0])
        relative_kps, mask = self._filter_3d_by_limb_length(relative_kps, mask)

        if np.sum(mask) < MIN_VALID_KEYPOINTS_FOR_PREDICTION: return None, None

        features = np.nan_to_num(relative_kps.flatten()).reshape(1, -1)
        try:
            pred_encoded = self.classifier_model.predict(features)
            label = self.label_encoder.inverse_transform(pred_encoded)[0]
            return label, {"kps_2d": keypoints_2d, "validity_mask": mask}
        except (IndexError, ValueError):
            return None, None

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
        if not (0 <= v_px < depth_map_mm.shape[0] and 0 <= u_px < depth_map_mm.shape[1]): return np.nan
        radius = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_start, y_end = max(0, v_px-radius), min(depth_map_mm.shape[0], v_px+radius+1)
        x_start, x_end = max(0, u_px-radius), min(depth_map_mm.shape[1], u_px+radius+1)
        neighborhood = depth_map_mm[y_start:y_end, x_start:x_end]
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
                filtered_points[i] = np.nan; validity_mask[i] = False
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
                            points[remove_idx] = np.nan; mask[remove_idx] = False; removed += 1
            if removed == 0: break
        return points, mask


class PoseDetectorNode:
    def __init__(self):
        rospy.init_node('pose_detector_node')

        model_type = rospy.get_param('~model_type', 'xgboost')
        
        models_dir = os.path.join(PACKAGE_ROOT_DIR, 'src', 'models', model_type)
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
        self.last_published_grasp = "none"

        self.grasp_pub = rospy.Publisher(GRASP_TYPE_TOPIC, String, queue_size=10)
        self.image_pub = rospy.Publisher("/miletic_dr/pose_detector_debug_img", Image, queue_size=1) # Debug tema

        rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
        # ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=2, slop=0.1)
        ts.registerCallback(self.synchronized_callback)

        rospy.loginfo("Pose Detector Node initialized and running.")

    def synchronized_callback(self, rgb_msg, depth_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Bridge conversion error: {e}")
            return
        
        # new_width = rgb_image.shape[1] // 2
        # new_height = rgb_image.shape[0] // 2
        # rgb_image_small = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        label, debug_data = self.classifier_logic.process_frame(rgb_image, depth_image)

        rospy.loginfo(f"Processed frame: label={label}")
        
        # --- Voting mehanizam ---
        if label is None:
            if self.last_published_grasp != "none":
                rospy.loginfo("Hand lost. Publishing 'none'.")
                self.grasp_pub.publish(String(data="none"))
                self.last_published_grasp = "none"
            self.vote_buffer.clear()
        else:
            self.vote_buffer.append(label)

        if len(self.vote_buffer) >= self.vote_buffer.maxlen:
            counts = Counter(self.vote_buffer)
            most_common, count = counts.most_common(1)[0]
            if count >= self.min_votes and most_common != self.last_published_grasp:
                self.grasp_pub.publish(String(data=most_common))
                self.last_published_grasp = most_common
                rospy.loginfo(f"Published new stable grasp: {most_common}")
        
        # debug viz
        if self.image_pub.get_num_connections() > 0 and debug_data is not None:
            self.publish_debug_image(rgb_image, debug_data)

    def publish_debug_image(self, image, debug_data):
        kps_2d = debug_data["kps_2d"]
        validity_mask = debug_data["validity_mask"]
        display_image = image.copy()
        for i, (u, v) in enumerate(kps_2d):
            if not np.isnan(u):
                color = (0, 255, 0) if validity_mask and validity_mask[i] else (0, 0, 255)
                cv2.circle(display_image, (int(u), int(v)), 4, color, -1)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_image, "bgr8"))

if __name__ == '__main__':
    try:
        PoseDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
