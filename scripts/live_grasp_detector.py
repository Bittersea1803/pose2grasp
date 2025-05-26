# File: scripts/live_grasp_detector.py
# Description: Performs real-time grasp detection using a trained model with 3D keypoint data.
# English comments.

import os
import sys
import time
import argparse

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters # For synchronizing image, depth, and camera_info
import joblib
from sklearn.impute import SimpleImputer
import pandas as pd

# --- Path Configuration ---
def get_project_root():
    """Gets the absolute path to the project's root directory (pose2grasp/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = get_project_root()
OPENPOSE_PYTHON_PATH = os.path.join(PROJECT_ROOT, "src", "pytorch-openpose")
MODELS_DIR_ROOT = os.path.join(PROJECT_ROOT, "models")

# --- OpenPose Import ---
try:
    if not os.path.isdir(OPENPOSE_PYTHON_PATH):
        raise ImportError(f"OpenPose Python path not found at: {OPENPOSE_PYTHON_PATH}")
    sys.path.append(OPENPOSE_PYTHON_PATH)
    from src.hand import Hand 
    from src import util as openpose_util
    rospy.loginfo("Successfully imported OpenPose modules.")
except ImportError as e:
    print(f"Fatal Error: Cannot import OpenPose from {OPENPOSE_PYTHON_PATH}.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration Constants ---
# Use the RGB topic that your models were trained on (likely rectified)
RGB_TOPIC = "/camera/rgb/image_rect_color" 
REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw" # Ensure this topic is active
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"
OPENPOSE_MODEL_FILE = os.path.join(OPENPOSE_PYTHON_PATH, "model", "hand_pose_model.pth")

# Processing parameters (should ideally match data_collect.py settings if used during collection)
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
VALID_DEPTH_THRESHOLD_MM = (400, 1500)
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0 # Max std dev in depth neighborhood
OUTLIER_XYZ_THRESHOLD_M = 0.5 # Max distance from wrist for a keypoint
APPLY_MEDIAN_FILTER_TO_DEPTH = True # Set to False if depth is already good or for speed
MEDIAN_FILTER_KERNEL_SIZE = 3

# For optimizing processing speed (less aggressive for 3D)
FRAME_PROCESSING_INTERVAL_MS = 200 # Process a frame roughly every X milliseconds (e.g., 200ms = 5 FPS)


class LiveGraspDetector:
    def __init__(self, model_type="xgboost"):
        rospy.init_node("live_grasp_detector_node", anonymous=True)
        self.bridge = CvBridge()
        
        self._latest_rgb_frame: Optional[np.ndarray] = None
        self._latest_depth_frame: Optional[np.ndarray] = None
        self._latest_camera_info: Optional[CameraInfo] = None
        self._fx: Optional[float] = None; self._fy: Optional[float] = None
        self._cx: Optional[float] = None; self._cy: Optional[float] = None
        self._camera_info_received = False
        
        self._last_processed_time = time.time()

        self.model_type = model_type.lower()
        self.model = None
        self.label_encoder = None
        self.imputer = None # For Random Forest
        self.feature_names = [f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')]

        # --- Load Model, Encoder, and Imputer (if RF) ---
        model_subdir = os.path.join(MODELS_DIR_ROOT, self.model_type)
        model_path = os.path.join(model_subdir, f"{self.model_type}_model.joblib" if self.model_type == "xgboost" else f"best_{self.model_type}_model.joblib")
        encoder_path = os.path.join(model_subdir, f"label_encoder_{self.model_type}.joblib")

        try:
            self.model = joblib.load(model_path)
            rospy.loginfo(f"{self.model_type.upper()} model loaded from {model_path}")
            self.label_encoder = joblib.load(encoder_path)
            rospy.loginfo(f"LabelEncoder loaded from {encoder_path}")

            if self.model_type == "random_forest":
                imputer_path = os.path.join(model_subdir, f"imputer_{self.model_type}.joblib")
                if os.path.exists(imputer_path):
                    self.imputer = joblib.load(imputer_path)
                    rospy.loginfo(f"Imputer loaded from {imputer_path}")
                else:
                    rospy.logwarn(f"Imputer for Random Forest not found at {imputer_path}. Creating a new one (median).")
                    self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                    # This new imputer is not fitted. It will be fitted on the first batch of data if NaNs are present,
                    # or you might need to save a fitted one from training.
                    # For simplicity, if NaNs appear and imputer isn't fitted, it might cause issues.
                    # Best to ensure the saved imputer from training is used.
        except FileNotFoundError as e:
            rospy.logfatal(f"Error loading model/encoder/imputer: {e}. Ensure files exist in '{model_subdir}'.")
            sys.exit(1)
        except Exception as e:
            rospy.logfatal(f"An unexpected error occurred: {e}")
            sys.exit(1)

        # --- Initialize OpenPose ---
        try:
            if not os.path.exists(OPENPOSE_MODEL_FILE):
                raise FileNotFoundError(f"OpenPose hand model not found: {OPENPOSE_MODEL_FILE}")
            self._hand_estimator = Hand(OPENPOSE_MODEL_FILE)
            self._openpose_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            rospy.loginfo(f"OpenPose initialized on device: {self._openpose_device}")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize OpenPose: {e}")
            sys.exit(1)

        # --- ROS Subscribers with Time Synchronizer ---
        self.rgb_sub = message_filters.Subscriber(RGB_TOPIC, Image)
        self.depth_sub = message_filters.Subscriber(REGISTERED_DEPTH_TOPIC, Image)
        self.info_sub = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1 # Allow 100ms slop
        )
        self.ts.registerCallback(self._synchronized_callback)
        
        rospy.loginfo(f"Synchronized subscribers initialized for topics:\n  RGB: {RGB_TOPIC}\n  Depth: {REGISTERED_DEPTH_TOPIC}\n  Info: {CAMERA_INFO_TOPIC}")
        rospy.loginfo("Live Grasp Detector node running. Waiting for data...")

    def _synchronized_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        current_time = time.time()
        if (current_time - self._last_processed_time) * 1000 < FRAME_PROCESSING_INTERVAL_MS:
            return # Skip frame
        self._last_processed_time = current_time

        if not self._camera_info_received:
            try:
                if len(info_msg.K) == 9:
                    self._fx, self._fy = info_msg.K[0], info_msg.K[4]
                    self._cx, self._cy = info_msg.K[2], info_msg.K[5]
                    self._camera_info_received = True
                    rospy.loginfo_once(f"Camera intrinsics received: fx={self._fx:.2f}, fy={self._fy:.2f}, cx={self._cx:.2f}, cy={self._cy:.2f}")
            except Exception as e:
                rospy.logerr_throttle(5, f"Error processing CameraInfo: {e}")
                return
        
        if not self._camera_info_received: # Still no intrinsics
            rospy.logwarn_throttle(5.0, "Waiting for valid CameraInfo to be processed...")
            return

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        self._process_frame_data(cv_rgb, cv_depth_mm)

    def _get_depth_from_neighborhood(self, depth_map_mm: np.ndarray, cx_px: float, cy_px: float, size: int) -> float:
        # (Copied and adapted from your data_collect.py - ensure consistency)
        if depth_map_mm is None: return np.nan
        radius = size // 2
        h, w = depth_map_mm.shape
        ix, iy = int(round(cx_px)), int(round(cy_px))

        if not (0 <= ix < w and 0 <= iy < h): return np.nan
        y_min, y_max = max(0, iy - radius), min(h, iy + radius + 1)
        x_min, x_max = max(0, ix - radius), min(w, ix + radius + 1)
        neighborhood = depth_map_mm[y_min:y_max, x_min:x_max]
        
        min_d, max_d_thresh = VALID_DEPTH_THRESHOLD_MM
        valid_depths = neighborhood[(neighborhood >= min_d) & (neighborhood <= max_d_thresh)]

        if valid_depths.size < max(1, (size*size)//4): return np.nan
        
        std_dev = np.std(valid_depths)
        if std_dev > DEPTH_STD_DEV_THRESHOLD_MM: return np.nan
        
        return float(np.median(valid_depths) / 1000.0) # Return in meters

    def _filter_3d_outliers(self, keypoints_3d_rel: np.ndarray) -> np.ndarray:
        # (Copied and adapted from your data_collect.py - ensure consistency)
        points_filtered = keypoints_3d_rel.copy()
        validity_mask = ~np.isnan(points_filtered).any(axis=1)
        if not validity_mask[0]: return points_filtered # Wrist must be valid

        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21): # Check points 1-20
            if validity_mask[i]:
                dist_sq = np.sum(points_filtered[i]**2)
                if dist_sq > max_dist_sq:
                    points_filtered[i] = np.nan
        return points_filtered

    def _process_frame_data(self, rgb_frame_bgr: np.ndarray, depth_frame_mm: np.ndarray):
        start_time_proc = time.time()
        try:
            # 1. Median Filter on Depth (Optional)
            depth_to_use = depth_frame_mm
            if APPLY_MEDIAN_FILTER_TO_DEPTH:
                ksize = MEDIAN_FILTER_KERNEL_SIZE
                if ksize % 2 == 0: ksize +=1
                depth_to_use = cv2.medianBlur(depth_frame_mm, ksize)

            # 2. OpenPose Hand Detection
            all_peaks_2d = self._hand_estimator(rgb_frame_bgr.copy())

            if not isinstance(all_peaks_2d, np.ndarray) or all_peaks_2d.ndim != 2 or all_peaks_2d.shape[0] == 0:
                sys.stdout.write("\rNo hand detected or invalid OpenPose output. FPS: ---")
                sys.stdout.flush()
                return

            # 3. Filter 2D peaks by confidence and prepare full 21-point array
            peaks_2d_for_projection = np.full((21, 2), np.nan, dtype=np.float32)
            num_confident_peaks = 0
            has_confidence = all_peaks_2d.shape[1] >= 3
            for i in range(min(21, all_peaks_2d.shape[0])):
                is_confident = (not has_confidence) or (all_peaks_2d[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD)
                if is_confident:
                    peaks_2d_for_projection[i, :] = all_peaks_2d[i, :2]
                    num_confident_peaks +=1
            
            if num_confident_peaks < MIN_VALID_KEYPOINTS_FOR_SAVE : # Check early if enough points
                sys.stdout.write(f"\rToo few confident 2D peaks: {num_confident_peaks}. FPS: ---")
                sys.stdout.flush()
                return

            # 4. Project to 3D (Camera Coordinates)
            keypoints_3d_cam = np.full((21, 3), np.nan, dtype=np.float32)
            for i in range(21):
                if not np.isnan(peaks_2d_for_projection[i, 0]): # If 2D point exists
                    x_px, y_px = float(peaks_2d_for_projection[i, 0]), float(peaks_2d_for_projection[i, 1])
                    z_m = self._get_depth_from_neighborhood(depth_to_use, x_px, y_px, DEPTH_NEIGHBORHOOD_SIZE)
                    if not np.isnan(z_m):
                        x_cam = (x_px - self._cx) * z_m / self._fx
                        y_cam = (y_px - self._cy) * z_m / self._fy
                        keypoints_3d_cam[i] = [x_cam, y_cam, z_m]
            
            # 5. Convert to Relative Coordinates (to wrist)
            keypoints_3d_rel = np.full_like(keypoints_3d_cam, np.nan)
            if not np.isnan(keypoints_3d_cam[0]).any(): # If wrist is valid
                wrist_3d = keypoints_3d_cam[0].copy()
                keypoints_3d_rel = keypoints_3d_cam - wrist_3d
                keypoints_3d_rel[0] = [0.0, 0.0, 0.0]
            else: # Wrist not detected in 3D
                sys.stdout.write(f"\rWrist not detected in 3D. FPS: ---")
                sys.stdout.flush()
                return
            
            # 6. Filter 3D outliers
            keypoints_3d_rel_filtered = self._filter_3d_outliers(keypoints_3d_rel)
            
            num_valid_final = np.sum(~np.isnan(keypoints_3d_rel_filtered).any(axis=1))
            if num_valid_final < MIN_VALID_KEYPOINTS_FOR_SAVE:
                sys.stdout.write(f"\rNot enough valid 3D points after filtering: {num_valid_final}. FPS: ---")
                sys.stdout.flush()
                return

            # 7. Prepare feature vector for model
            feature_vector_list = []
            for i in range(21):
                feature_vector_list.extend(keypoints_3d_rel_filtered[i])
            
            features_df = pd.DataFrame([feature_vector_list], columns=self.feature_names)

            # 8. Handle NaNs if Random Forest (using the loaded, fitted imputer)
            if self.model_type == "random_forest" and self.imputer:
                if features_df.isnull().sum().sum() > 0:
                    try:
                        features_df = pd.DataFrame(self.imputer.transform(features_df), columns=self.feature_names)
                    except ValueError as e_impute: # Handle case where imputer might not have seen all NaN patterns
                        rospy.logwarn_throttle(10, f"Imputer error: {e_impute}. Trying to fill remaining NaNs with 0 for RF.")
                        features_df = features_df.fillna(0) # Fallback
            elif self.model_type == "xgboost" and features_df.isnull().sum().sum() > 0:
                # XGBoost handles NaNs, but if you want to fill them to avoid potential issues or for consistency:
                # features_df = features_df.fillna(0) # Example: fill with 0 for XGBoost
                pass # Let XGBoost handle it by default

            # 9. Predict
            prediction_numeric = self.model.predict(features_df)
            predicted_label_str = self.label_encoder.inverse_transform(prediction_numeric)[0]
            
            end_time_proc = time.time()
            fps_proc = 1.0 / (end_time_proc - start_time_proc) if (end_time_proc - start_time_proc) > 0 else float('inf')
            
            sys.stdout.write(f"\rPredicted Grasp: {predicted_label_str.upper():<10} | Valid 3D pts: {num_valid_final:<2} | FPS: {fps_proc:.1f}   ")
            sys.stdout.flush()

        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in frame processing: {e}")
            # import traceback # Uncomment for detailed debugging
            # rospy.logerr_throttle(5.0, traceback.format_exc())


    def run(self):
        rospy.spin()
        rospy.loginfo("Live Grasp Detector node shutting down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live grasp detection using a trained model.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="xgboost", 
        choices=["xgboost", "random_forest"],
        help="Specify the model type to use: 'xgboost' or 'random_forest'. Default: xgboost"
    )
    args = parser.parse_args(rospy.myargv()[1:])

    try:
        detector = LiveGraspDetector(model_type=args.model)
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Live detector node interrupted.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in live detector: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rospy.loginfo("Live grasp detector node finished.")