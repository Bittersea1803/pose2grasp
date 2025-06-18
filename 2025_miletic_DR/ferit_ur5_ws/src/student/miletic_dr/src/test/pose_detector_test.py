#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# works with lab cam

import os
import sys
import threading
import queue
from collections import deque, Counter
import cv2
import numpy as np
import torch
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image as PILImage, ImageTk
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(SCRIPT_DIR, '..')
    PACKAGE_ROOT_DIR = os.path.join(SRC_DIR, '..')
    sys.path.append(SRC_DIR)

    OPENPOSE_PATH = os.path.join(PACKAGE_ROOT_DIR, 'pytorch-openpose')
    sys.path.append(OPENPOSE_PATH)
    from src.hand import Hand
    rospy.loginfo("Successfully configured paths and imported modules.")
except ImportError as e:
    rospy.logfatal(f"Could not import a required module. Check paths. Error: {e}")
    sys.exit(1)


# CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
# CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
# CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"

VALID_DEPTH_THRESHOLD_MM = (400, 1500)
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
VOTING_WINDOW_SIZE = 15
VOTING_AGREEMENT_PERCENTAGE = 0.7
MIN_VALID_KEYPOINTS_FOR_PREDICTION = 15
OUTLIER_XYZ_THRESHOLD_M = 0.25
MAX_LIMB_LENGTH_M = 0.10
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
MEDIAN_FILTER_KERNEL_SIZE = 3


class LiveGraspClassifierApp:
    def __init__(self, model_type="xgboost"):
        rospy.init_node("live_grasp_classifier_node", anonymous=True)
        self.bridge = CvBridge()
        self.is_shutting_down = False
        self._data_access_lock = threading.Lock()
        
        self.is_classifying_continuously = False
        self.processing_thread = None
        
        self.last_known_kps = None
        self.last_known_validity_mask = None

        self._vote_buffer = deque(maxlen=VOTING_WINDOW_SIZE)
        self._min_votes_for_decision = int(VOTING_WINDOW_SIZE * VOTING_AGREEMENT_PERCENTAGE)
        
        try:
            self._load_models(model_type)
        except Exception as e:
            rospy.logfatal(f"Fatal error during model loading: {e}")
            sys.exit(1)

        self.root = tk.Tk()
        self.root.title("Live Grasp Classifier")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._gui_queue = queue.Queue()
        self._setup_gui()
        
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self._latest_rgb = None
        self._latest_depth = None
        self._initialize_ros_subscriptions()
        
        self._process_gui_queue()
        rospy.loginfo("Application fully initialized.")

    def _load_models(self, model_type):
        openpose_model_path = os.path.join(OPENPOSE_PATH, "model", "hand_pose_model.pth")
        self._hand_estimator = Hand(openpose_model_path)
        models_dir = os.path.join(SRC_DIR, 'models', model_type)
        model_path = os.path.join(models_dir, f'{model_type}_model.joblib')
        encoder_path = os.path.join(models_dir, f'label_encoder_{model_type}.joblib')
        self.classifier_model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(main_frame, text="Waiting for video...")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.classify_button = ttk.Button(control_frame, text="Start Live Classification", command=self._toggle_live_classification)
        self.classify_button.pack(fill=tk.X, ipady=10, padx=5)
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=10, padx=5)
        self.raw_pred_var = tk.StringVar(value="Raw: --")
        ttk.Label(info_frame, textvariable=self.raw_pred_var).pack(anchor=tk.W)
        self.confidence_var = tk.StringVar(value="Confidence: --")
        ttk.Label(info_frame, textvariable=self.confidence_var).pack(anchor=tk.W)
        self.voter_status_var = tk.StringVar(value="Voter: --")
        ttk.Label(info_frame, textvariable=self.voter_status_var).pack(anchor=tk.W)
        self.stable_prediction_var = tk.StringVar(value="--")
        result_label_font = ("Helvetica", 20, "bold")
        result_label = ttk.Label(info_frame, textvariable=self.stable_prediction_var, foreground="blue", anchor="center", font=result_label_font)
        result_label.pack(fill=tk.X, pady=10)
        self.status_var = tk.StringVar(value="Ready. Press 'Start' to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _initialize_ros_subscriptions(self):
        try:
            info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
            self.fx, self.fy, self.cx, self.cy = info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5]
            rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
            depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
            ts.registerCallback(self.ros_callback)
        except rospy.ROSException as e:
            self.on_close()

    def ros_callback(self, rgb_msg, depth_msg):
        with self._data_access_lock:
            self._latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self._latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

    def _toggle_live_classification(self):
        if self.is_classifying_continuously:
            self.is_classifying_continuously = False
            self.classify_button.config(text="Start Live Classification")
            self.status_var.set("Stopped. Press 'Start' to resume.")
            self.last_known_kps = None
            self.last_known_validity_mask = None
        else:
            self.is_classifying_continuously = True
            self._vote_buffer.clear()
            self.classify_button.config(text="Stop Live Classification")
            self.status_var.set("Running live classification...")
            self.processing_thread = threading.Thread(target=self._continuous_classification_loop, daemon=True)
            self.processing_thread.start()

    def _continuous_classification_loop(self):
        while self.is_classifying_continuously and not rospy.is_shutdown():
            with self._data_access_lock:
                if self._latest_rgb is None:
                    rospy.sleep(0.1)
                    continue
                rgb_image = self._latest_rgb.copy()
                depth_image = self._latest_depth.copy()
            
            try:
                depth_image_filtered = cv2.medianBlur(depth_image, MEDIAN_FILTER_KERNEL_SIZE)
                all_peaks_2d = self._hand_estimator(rgb_image)
                if all_peaks_2d is None: raise ValueError("Hand not detected")

                has_confidence = all_peaks_2d.shape[1] == 3
                keypoints_2d = np.full((21, 2), np.nan, dtype=np.float32)
                for i in range(min(21, all_peaks_2d.shape[0])):
                    if (has_confidence and all_peaks_2d[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD) or (not has_confidence):
                        keypoints_2d[i] = all_peaks_2d[i, :2]
                
                keypoints_3d = self._project_points_to_3d(keypoints_2d, depth_image_filtered)
                if np.isnan(keypoints_3d[0]).any(): raise ValueError("Wrist not in 3D")

                relative_kps = keypoints_3d - keypoints_3d[0]
                relative_kps, validity_mask = self._filter_3d_outliers(relative_kps)
                relative_kps, validity_mask = self._filter_3d_by_limb_length(relative_kps, validity_mask)

                if np.sum(validity_mask) < MIN_VALID_KEYPOINTS_FOR_PREDICTION:
                    raise ValueError(f"Filtered points ({np.sum(validity_mask)}) below threshold")

                features = np.nan_to_num(relative_kps.flatten().reshape(1, -1))
                pred = self.classifier_model.predict(features)
                probs = self.classifier_model.predict_proba(features)[0]
                
                result_payload = {
                    "kps_2d": keypoints_2d, "validity_mask": validity_mask,
                    "raw_label": self.label_encoder.inverse_transform(pred)[0],
                    "confidence": probs[pred[0]],
                }
                self._gui_queue.put(("update_data", result_payload))

            except Exception as e:
                self._gui_queue.put(("clear_data", str(e)))
            
            rospy.sleep(0.1)
    
    def _filter_3d_outliers(self, keypoints_3d_relative):
        filtered_points = keypoints_3d_relative.copy()
        validity_mask = ~np.isnan(filtered_points).any(axis=1)
        if not validity_mask[0]: return filtered_points, validity_mask.tolist()
        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21):
            if validity_mask[i] and np.sum(filtered_points[i]**2) > max_dist_sq:
                filtered_points[i] = np.nan
                validity_mask[i] = False
        return filtered_points, validity_mask.tolist()

    def _filter_3d_by_limb_length(self, keypoints_3d_rel, validity_mask):
        points = keypoints_3d_rel.copy()
        mask = list(validity_mask)
        for _ in range(3):
            removed_in_pass = 0
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if mask[p1_idx] and mask[p2_idx]:
                    if np.sum((points[p1_idx] - points[p2_idx])**2) > MAX_LIMB_LENGTH_M**2:
                        d1_sq, d2_sq = np.sum(points[p1_idx]**2), np.sum(points[p2_idx]**2)
                        idx_to_remove = p1_idx if d1_sq > d2_sq else p2_idx
                        if mask[idx_to_remove]:
                            points[idx_to_remove] = np.nan
                            mask[idx_to_remove] = False
                            removed_in_pass += 1
            if removed_in_pass == 0: break
        return points, mask

    def _project_points_to_3d(self, keypoints_2d, depth_map):
        keypoints_3d = np.full((21, 3), np.nan, dtype=np.float32)
        for i, (u, v) in enumerate(keypoints_2d):
            if not np.isnan(u):
                z = self.get_robust_depth(depth_map, int(u), int(v))
                if not np.isnan(z):
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    keypoints_3d[i] = [x, y, z]
        return keypoints_3d
    
    def get_robust_depth(self, depth_map_mm, u_px, v_px):
        if not (0 <= v_px < depth_map_mm.shape[0] and 0 <= u_px < depth_map_mm.shape[1]):
            return np.nan
        
        radius = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_start, y_end = max(0, v_px - radius), min(depth_map_mm.shape[0], v_px + radius + 1)
        x_start, x_end = max(0, u_px - radius), min(depth_map_mm.shape[1], u_px + radius + 1)
        
        neighborhood = depth_map_mm[y_start:y_end, x_start:x_end]
        valid_depths = neighborhood[(neighborhood >= VALID_DEPTH_THRESHOLD_MM[0]) & (neighborhood <= VALID_DEPTH_THRESHOLD_MM[1])]
        if valid_depths.size < max(1, (DEPTH_NEIGHBORHOOD_SIZE**2) // 4): return np.nan
        if np.std(valid_depths) > DEPTH_STD_DEV_THRESHOLD_MM: return np.nan
        
        return float(np.median(valid_depths)) / 1000.0

    def _process_gui_queue(self):
        with self._data_access_lock:
            if self._latest_rgb is not None:
                self.update_video_display(self._latest_rgb, self.last_known_kps, self.last_known_validity_mask)
        try:
            task, payload = self._gui_queue.get_nowait()
            if task == "update_data":
                self.last_known_kps = payload["kps_2d"]
                self.last_known_validity_mask = payload["validity_mask"]
                self.update_text_labels(payload)
            elif task == "clear_data":
                self.last_known_kps = None
                self.last_known_validity_mask = None
                self.clear_text_labels(str(payload))
        except queue.Empty:
            pass
        finally:
            if not self.is_shutting_down:
                self.root.after(40, self._process_gui_queue)

    def update_text_labels(self, payload):
        raw_label = payload['raw_label']
        self._vote_buffer.append(raw_label.lower())
        stable_label, voter_status_text = self._get_voter_status()
        self.raw_pred_var.set(f"Raw: {raw_label}")
        self.confidence_var.set(f"Confidence: {payload['confidence']:.2%}")
        self.voter_status_var.set(f"Voter: {voter_status_text}")
        self.stable_prediction_var.set(stable_label)
        self.status_var.set("Running live classification...")
        
    def clear_text_labels(self, status_msg):
        self.raw_pred_var.set("Raw: --")
        self.confidence_var.set("Confidence: --")
        self.voter_status_var.set("Voter: --")
        self.stable_prediction_var.set("--")
        self.status_var.set(f"Status: {status_msg}")

    def update_video_display(self, image_bgr, keypoints_2d, validity_mask):
        display_image = image_bgr.copy()
        if keypoints_2d is not None:
            for i, (u, v) in enumerate(keypoints_2d):
                if not np.isnan(u):
                    color = (0, 255, 0) if validity_mask and validity_mask[i] else (0, 0, 255)
                    cv2.circle(display_image, (int(u), int(v)), 4, color, -1)
        
        img_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        max_h, max_w = 480, 640
        if h > max_h or w > max_w:
            scale = min(max_h / h, max_w / w)
            img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
        img_pil = PILImage.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

    def _get_voter_status(self):
        if len(self._vote_buffer) < VOTING_WINDOW_SIZE:
            return "--", f"Collecting... ({len(self._vote_buffer)}/{VOTING_WINDOW_SIZE})"
        counts = Counter(self._vote_buffer)
        most_common, num_votes = counts.most_common(1)[0]
        if num_votes >= self._min_votes_for_decision:
            return most_common.upper(), f"Stable ({num_votes}/{VOTING_WINDOW_SIZE})"
        else:
            return "--", f"Unstable ({num_votes}/{VOTING_WINDOW_SIZE} for '{most_common}')"

    def on_close(self):
        self.is_shutting_down = True
        rospy.signal_shutdown("GUI Closed")
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    try:
        app = LiveGraspClassifierApp()
        print("Starting Live Grasp Classifier GUI...")
        app.run()
        print("Live Grasp Classifier GUI closed.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in main: {e}")