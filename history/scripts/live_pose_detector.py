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
import message_filters 
import joblib
from std_msgs.msg import String
from collections import deque, Counter
from sklearn.impute import SimpleImputer
import pandas as pd

# --- Path Configuration ---
def get_project_root():
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
    rospy.loginfo("Successfully imported OpenPose modules.")
except ImportError as e:
    print(f"Fatal Error: Cannot import OpenPose from {OPENPOSE_PYTHON_PATH}.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration Constants ---
RGB_TOPIC = "/camera/rgb/image_rect_color" 
REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw" 
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"
OPENPOSE_MODEL_FILE = os.path.join(OPENPOSE_PYTHON_PATH, "model", "hand_pose_model.pth")

OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
VALID_DEPTH_THRESHOLD_MM = (400, 1500)
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
OUTLIER_XYZ_THRESHOLD_M = 0.25 
APPLY_MEDIAN_FILTER_TO_DEPTH = True
MEDIAN_FILTER_KERNEL_SIZE = 3
MIN_VALID_KEYPOINTS_FOR_SAVE = 13 
MAX_LIMB_LENGTH_M = 0.10          

FRAME_PROCESSING_INTERVAL_MS = 100 # Reduced for potentially faster voting updates

# --- Voting Configuration ---
VOTING_WINDOW_SIZE = 10          # Number of high-confidence predictions to consider in the sliding window
VOTING_CONFIDENCE_THRESHOLD = 0.60 # Minimum probability for a prediction to be considered a vote
MIN_VOTE_AGREEMENT_PERCENTAGE = 0.7 # Pct of votes in window that must agree for a stable decision (e.g., 0.7 for 70%)
GRASP_DECISION_TOPIC = "/pose2grasp/grasp_type" # Topic to publish the stable, voted grasp decision

HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]

from typing import Optional, List, Tuple 

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
        self.imputer = None
        self.feature_names = [f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')]

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
                    rospy.logfatal(f"Imputer for Random Forest not found at {imputer_path}. This is required for Random Forest. Please ensure it's generated during training.")
                    sys.exit(1) # Exit if imputer is missing for RF
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
            slop=0.1
        )
        self.ts.registerCallback(self._synchronized_callback)
        
        rospy.loginfo(f"Synchronized subscribers initialized for topics:\n  RGB: {RGB_TOPIC}\n  Depth: {REGISTERED_DEPTH_TOPIC}\n  Info: {CAMERA_INFO_TOPIC}")
        rospy.loginfo("Live Grasp Detector node running. Waiting for data...")

        # --- Voting Mechanism ---
        self.VOTING_WINDOW_SIZE = VOTING_WINDOW_SIZE
        self.VOTING_CONFIDENCE_THRESHOLD = VOTING_CONFIDENCE_THRESHOLD
        self.MIN_VOTE_AGREEMENT_PERCENTAGE = MIN_VOTE_AGREEMENT_PERCENTAGE
        self._min_votes_for_decision = max(1, int(self.VOTING_WINDOW_SIZE * self.MIN_VOTE_AGREEMENT_PERCENTAGE))

        self._vote_buffer = deque(maxlen=self.VOTING_WINDOW_SIZE)
        self._current_stable_prediction: Optional[str] = None
        self.grasp_decision_publisher = rospy.Publisher(GRASP_DECISION_TOPIC, String, queue_size=10)
        
        rospy.loginfo(f"Voting mechanism initialized: Window={self.VOTING_WINDOW_SIZE}, ConfThresh={self.VOTING_CONFIDENCE_THRESHOLD}, VoteAgreePct={self.MIN_VOTE_AGREEMENT_PERCENTAGE} (MinVotes={self._min_votes_for_decision})")
        rospy.loginfo(f"Publishing voted grasp decisions to: {GRASP_DECISION_TOPIC}")


    def _synchronized_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        current_time = time.time()
        if (current_time - self._last_processed_time) * 1000 < FRAME_PROCESSING_INTERVAL_MS:
            return
        self._last_processed_time = current_time

        if not self._camera_info_received:
            try:
                # Check for K attribute, 9 elements, and positive fx, fy
                if hasattr(info_msg, 'K') and len(info_msg.K) == 9 and \
                   info_msg.K[0] > 0 and info_msg.K[4] > 0:
                    self._fx, self._fy = info_msg.K[0], info_msg.K[4]
                    self._cx, self._cy = info_msg.K[2], info_msg.K[5]
                    self._camera_info_received = True
                    rospy.loginfo_once(f"Camera intrinsics received: fx={self._fx:.2f}, fy={self._fy:.2f}, cx={self._cx:.2f}, cy={self._cy:.2f}")
                # else: # Optionally, log if K is present but not valid yet
                    # rospy.logwarn_throttle(5.0, "CameraInfo K matrix received but fx/fy are not positive yet.")
            except Exception as e:
                rospy.logerr_throttle(5, f"Error processing CameraInfo: {e}")
                return
        
        if not self._camera_info_received: 
            rospy.logwarn_throttle(5.0, "Waiting for valid CameraInfo to be processed...")
            return

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        self._process_frame_data(cv_rgb, cv_depth_mm)

    def _get_depth_from_neighborhood(self, depth_map_mm: np.ndarray, cx_px: float, cy_px: float, size: int, keypoint_idx_for_log: Optional[int] = None) -> float:
        if depth_map_mm is None: return np.nan
        radius = size // 2
        h, w = depth_map_mm.shape
        ix, iy = int(round(cx_px)), int(round(cy_px))
        log_prefix = f"Wrist (0) depth: " if keypoint_idx_for_log == 0 else (f"KP{keypoint_idx_for_log} depth: " if keypoint_idx_for_log is not None else "Depth: ")

        if not (0 <= ix < w and 0 <= iy < h):
            if keypoint_idx_for_log == 0: # Log only for wrist to reduce noise
                rospy.logwarn_throttle(5, f"{log_prefix}Coords ({ix},{iy}) out of bounds ({w},{h}).")
            return np.nan

        y_min, y_max = max(0, iy - radius), min(h, iy + radius + 1)
        x_min, x_max = max(0, ix - radius), min(w, ix + radius + 1)
        neighborhood = depth_map_mm[y_min:y_max, x_min:x_max]
        
        min_d, max_d_thresh = VALID_DEPTH_THRESHOLD_MM
        valid_depths = neighborhood[(neighborhood >= min_d) & (neighborhood <= max_d_thresh)]

        if valid_depths.size < max(1, (size*size)//4):
            if keypoint_idx_for_log == 0: # Log only for wrist
                 rospy.logwarn_throttle(5, f"{log_prefix}Not enough valid depth pixels in neighborhood. Found {valid_depths.size}, need {max(1, (size*size)//4)} at ({cx_px:.1f},{cy_px:.1f}). Values: {valid_depths[:5]}") # Show some values
            return np.nan
        
        std_dev = np.std(valid_depths)
        if std_dev > DEPTH_STD_DEV_THRESHOLD_MM:
            if keypoint_idx_for_log == 0: # Log only for wrist
                rospy.logwarn_throttle(5, f"{log_prefix}Depth std dev too high ({std_dev:.1f}mm > {DEPTH_STD_DEV_THRESHOLD_MM}mm) at ({cx_px:.1f},{cy_px:.1f}).")
            return np.nan
        if keypoint_idx_for_log == 0: rospy.loginfo_throttle(2, f"{log_prefix}Depth median {np.median(valid_depths):.1f}mm, std {std_dev:.1f}mm for coords ({cx_px:.1f},{cy_px:.1f})")
        return float(np.median(valid_depths) / 1000.0) # Return in meters

    def _filter_3d_outliers(self, keypoints_3d_rel: np.ndarray) -> np.ndarray:
        points_filtered = keypoints_3d_rel.copy()
        validity_mask = ~np.isnan(points_filtered).any(axis=1)
        if not validity_mask[0]: return points_filtered

        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21):
            if validity_mask[i]:
                dist_sq = np.sum(points_filtered[i]**2)
                if dist_sq > max_dist_sq:
                    points_filtered[i] = np.nan
        return points_filtered

    def _filter_3d_by_limb_length(self, keypoints_3d_rel: np.ndarray, validity_mask: List[bool]) -> Tuple[np.ndarray, List[bool]]:
        """
        Filters 3D keypoints by checking limb lengths.
        Adapted from data_collect.py for consistency.
        """
        points_filtered = keypoints_3d_rel.copy()
        new_validity_mask = list(validity_mask)

        MAX_ITERATIONS = 5 # Safety break for the while loop
        for iteration in range(MAX_ITERATIONS):
            num_removed_in_pass = 0
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if not (0 <= p1_idx < len(new_validity_mask) and 0 <= p2_idx < len(new_validity_mask)):
                    continue

                if new_validity_mask[p1_idx] and new_validity_mask[p2_idx]:
                    p1 = points_filtered[p1_idx]
                    p2 = points_filtered[p2_idx]

                    if np.isnan(p1).any() or np.isnan(p2).any():
                        continue

                    dist_sq = np.sum((p1 - p2)**2)

                    if dist_sq > MAX_LIMB_LENGTH_M**2:
                        dist_p1_sq_from_origin = np.sum(p1**2) # Origin is wrist for relative coords
                        dist_p2_sq_from_origin = np.sum(p2**2)

                        idx_to_invalidate = p1_idx if dist_p1_sq_from_origin > dist_p2_sq_from_origin else p2_idx
                        if new_validity_mask[idx_to_invalidate]:
                            points_filtered[idx_to_invalidate] = np.nan
                            new_validity_mask[idx_to_invalidate] = False
                            num_removed_in_pass += 1
            
            if num_removed_in_pass == 0:
                break 
        
        if iteration == MAX_ITERATIONS - 1 and num_removed_in_pass > 0:
            rospy.logwarn_throttle(10, f"Limb length filter reached max iterations ({MAX_ITERATIONS}) and still making changes.")
                        
        return points_filtered, new_validity_mask

    def _process_frame_data(self, rgb_frame_bgr: np.ndarray, depth_frame_mm: np.ndarray):
        start_time_proc = time.time()
        try:
            # 1. Median Filter on Depth
            depth_to_use = depth_frame_mm
            if APPLY_MEDIAN_FILTER_TO_DEPTH:
                ksize = MEDIAN_FILTER_KERNEL_SIZE
                if ksize % 2 == 0: ksize +=1
                depth_to_use = cv2.medianBlur(depth_frame_mm, ksize)

            # 2. OpenPose Hand Detection
            all_peaks_2d = self._hand_estimator(rgb_frame_bgr.copy())
            if isinstance(all_peaks_2d, np.ndarray) and all_peaks_2d.shape[0] > 0:
                wrist_peak_raw = all_peaks_2d[0] if all_peaks_2d.shape[0] > 0 else "N/A"
                rospy.loginfo_throttle(2, f"OpenPose raw wrist (0) peak: {wrist_peak_raw}")
            else:
                rospy.loginfo_throttle(2, f"OpenPose raw output: {all_peaks_2d}")

            if not isinstance(all_peaks_2d, np.ndarray) or all_peaks_2d.ndim != 2 or all_peaks_2d.shape[0] == 0:
                sys.stdout.write("\rNo hand detected. RawFPS: --- Voted: Waiting...")
                sys.stdout.flush()
                return

            # Ensure all_peaks_2d has at least 3 columns (x, y, confidence)
            has_confidence = all_peaks_2d.shape[1] >= 3

            # 3. Filter 2D peaks by confidence and prepare full 21-point array
            peaks_2d_for_projection = np.full((21, 2), np.nan, dtype=np.float32)
            num_confident_peaks = 0
            has_confidence = all_peaks_2d.shape[1] >= 3
            for i in range(min(21, all_peaks_2d.shape[0])):
                is_confident = (not has_confidence) or (all_peaks_2d[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD)
                if is_confident:
                    peaks_2d_for_projection[i, :] = all_peaks_2d[i, :2]
                    num_confident_peaks +=1
                if i == 0: # Specifically log wrist confidence
                    wrist_conf = all_peaks_2d[i, 2] if has_confidence else "N/A (no conf column)"
                    rospy.loginfo_throttle(2, f"Wrist (0) 2D coords: {all_peaks_2d[i, :2]}, Conf: {wrist_conf}, Passes conf filter: {is_confident}")
            
            rospy.loginfo_throttle(2, f"Wrist (0) 2D after confidence filter: {peaks_2d_for_projection[0]}")

            if num_confident_peaks < MIN_VALID_KEYPOINTS_FOR_SAVE:
                sys.stdout.write(f"\rLow 2D conf peaks: {num_confident_peaks}. RawFPS: --- Voted: Waiting...")
                sys.stdout.flush()
                return

            # 4. Project to 3D
            keypoints_3d_cam = np.full((21, 3), np.nan, dtype=np.float32)
            for i in range(21):
                if not np.isnan(peaks_2d_for_projection[i, 0]): # If 2D point exists
                    x_px, y_px = float(peaks_2d_for_projection[i, 0]), float(peaks_2d_for_projection[i, 1])
                    z_m = self._get_depth_from_neighborhood(depth_to_use, x_px, y_px, DEPTH_NEIGHBORHOOD_SIZE, keypoint_idx_for_log=i)
                    if not np.isnan(z_m):
                        x_cam = (x_px - self._cx) * z_m / self._fx
                        y_cam = (y_px - self._cy) * z_m / self._fy
                        keypoints_3d_cam[i] = [x_cam, y_cam, z_m]
            
            rospy.loginfo_throttle(2, f"Wrist (0) 3D cam coords before relative conversion: {keypoints_3d_cam[0]}")

            # 5. Convert to Relative Coordinates (to wrist)
            keypoints_3d_rel = np.full_like(keypoints_3d_cam, np.nan)
            if not np.isnan(keypoints_3d_cam[0]).any(): # If wrist is valid
                wrist_3d = keypoints_3d_cam[0].copy()
                keypoints_3d_rel = keypoints_3d_cam - wrist_3d
                keypoints_3d_rel[0] = [0.0, 0.0, 0.0]
            else: # Wrist not detected in 3D
                sys.stdout.write(f"\rWrist not in 3D. RawFPS: --- Voted: Waiting...")
                sys.stdout.flush()
                return
            
            # 6. Filter 3D outliers (distance from wrist)
            keypoints_3d_rel_filtered_outliers = self._filter_3d_outliers(keypoints_3d_rel)

            # 6b. Filter 3D by limb length (for consistency with data_collect.py)
            # Create an initial validity mask based on the outlier filter's results
            current_validity_mask_after_outliers = (~np.isnan(keypoints_3d_rel_filtered_outliers).any(axis=1)).tolist()
            
            keypoints_3d_rel_final_filtered, final_validity_mask_list = self._filter_3d_by_limb_length(
                keypoints_3d_rel_filtered_outliers, current_validity_mask_after_outliers
            )
            num_valid_final = np.sum(final_validity_mask_list)

            if num_valid_final < MIN_VALID_KEYPOINTS_FOR_SAVE:
                sys.stdout.write(f"\rLow 3D valid pts: {num_valid_final}. RawFPS: --- Voted: Waiting...")
                sys.stdout.flush()
                return

            # 7. Prepare feature vector for model
            feature_vector_list = []
            for i in range(21):
                feature_vector_list.extend(keypoints_3d_rel_final_filtered[i])
            
            features_df = pd.DataFrame([feature_vector_list], columns=self.feature_names)

            # 8. Handle NaNs if Random Forest (using imputer)
            if self.model_type == "random_forest" and self.imputer:
                if features_df.isnull().sum().sum() > 0:
                    try:
                        features_df = pd.DataFrame(self.imputer.transform(features_df), columns=self.feature_names)
                    except ValueError as e_impute: # Handle case where imputer might not have seen all NaN patterns
                        rospy.logwarn_throttle(10, f"Imputer error: {e_impute}. Trying to fill remaining NaNs with 0 for RF.")
                        features_df = features_df.fillna(0) # Fallback
            elif self.model_type == "xgboost" and features_df.isnull().sum().sum() > 0:
                # features_df = features_df.fillna(0)
                pass

            # 9. Predict
            # Get raw prediction and its confidence
            raw_prediction_numeric = self.model.predict(features_df)
            current_raw_prediction_label = self.label_encoder.inverse_transform(raw_prediction_numeric)[0]
            
            probabilities = self.model.predict_proba(features_df)[0]
            # The predicted class index from predict() corresponds to the max probability
            confidence_of_predicted_class = probabilities[raw_prediction_numeric[0]]

            # --- Voting Logic ---
            voted_grasp_status_for_display = "Waiting..."

            if confidence_of_predicted_class >= self.VOTING_CONFIDENCE_THRESHOLD:
                self._vote_buffer.append(current_raw_prediction_label)
            # else:
                # rospy.logdebug_throttle(2, f"Raw pred '{current_raw_prediction_label}' conf {confidence_of_predicted_class:.2f} too low. Vote not added.")

            if len(self._vote_buffer) == self.VOTING_WINDOW_SIZE: # Buffer is full, time to evaluate
                vote_counts = Counter(self._vote_buffer)
                # most_common_vote can be None if buffer is empty, but it won't be if len is VOTING_WINDOW_SIZE > 0
                most_common_vote, num_most_common = vote_counts.most_common(1)[0] 

                if num_most_common >= self._min_votes_for_decision:
                    voted_grasp_status_for_display = f"Stable: {most_common_vote} ({num_most_common}/{self.VOTING_WINDOW_SIZE})"
                    if self._current_stable_prediction != most_common_vote:
                        self._current_stable_prediction = most_common_vote
                        self.grasp_decision_publisher.publish(String(data=most_common_vote))
                        rospy.loginfo(f"VOTED GRASP PUBLISHED: {most_common_vote} (Votes: {num_most_common}/{len(self._vote_buffer)})")
                else: # Not enough consensus
                    voted_grasp_status_for_display = f"Unstable ({most_common_vote}:{num_most_common})"
                    if self._current_stable_prediction is not None:
                        rospy.loginfo(f"Voted grasp consensus lost. Was: {self._current_stable_prediction}. Current top: {most_common_vote} ({num_most_common} votes). Buffer: {list(self._vote_buffer)}")
                        self.grasp_decision_publisher.publish(String(data="unknown")) # Publish unknown
                        self._current_stable_prediction = None
            elif len(self._vote_buffer) > 0 :
                 voted_grasp_status_for_display = f"Collecting ({len(self._vote_buffer)}/{self.VOTING_WINDOW_SIZE})"
            else:
                voted_grasp_status_for_display = "No high-conf votes yet"

            # --- Timing and Display ---
            end_time_proc = time.time()
            fps_proc = 1.0 / (end_time_proc - start_time_proc) if (end_time_proc - start_time_proc) > 0 else float('inf')
            
            display_msg = f"\rRaw: {current_raw_prediction_label.upper():<10} ({confidence_of_predicted_class:.2f}) | Voted: {voted_grasp_status_for_display:<30} | 3Dpts: {num_valid_final:<2} | FPS: {fps_proc:.1f}  "
            sys.stdout.write(display_msg)
            sys.stdout.flush()

        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in frame processing: {e}")
            # import traceback 
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