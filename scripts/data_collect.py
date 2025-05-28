import os
import sys
import queue
import threading
import csv
import datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image as PILImage, ImageTk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Path Configuration ---
def get_project_root():
    """Finds the project's root directory automatically."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = get_project_root()
OPENPOSE_PYTHON_PATH = os.path.join(PROJECT_ROOT, "src", "pytorch-openpose")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OPENPOSE_MODEL_DIR = os.path.join(OPENPOSE_PYTHON_PATH, "model")

# --- OpenPose Import ---
try:
    if not os.path.isdir(OPENPOSE_PYTHON_PATH):
        raise ImportError(f"OpenPose Python path not found at: {OPENPOSE_PYTHON_PATH}")
    sys.path.append(OPENPOSE_PYTHON_PATH)
    from src.hand import Hand 
    rospy.loginfo("Successfully imported OpenPose modules.")
except ImportError as e:
    rospy.logfatal(f"Cannot import OpenPose. Check path and installation. Error: {e}")
    sys.exit(1)

# --- Main Constants ---
RGB_TOPIC_RAW = "/camera/rgb/image_color"
RGB_TOPIC_RECT_COLOR = "/camera/rgb/image_rect_color"
RGB_TOPIC_OPTIONS = {
    "Rectified Color (`image_rect_color`)": RGB_TOPIC_RECT_COLOR,
    "Raw Color (`image_color`)": RGB_TOPIC_RAW
}

REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

CSV_FILENAME = "collected_hand_poses.csv" # Unified CSV filename
CSV_FULL_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

VALID_DEPTH_THRESHOLD_MM = (400, 1500)
MIN_VALID_KEYPOINTS_FOR_SAVE = 18 
DISPLAY_MAX_WIDTH = 320
DISPLAY_MAX_HEIGHT = 240
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
OUTLIER_XYZ_THRESHOLD_M = 0.25 # Changed from 0.5
MEDIAN_FILTER_KERNEL_SIZE = 3
POSE_LABELS = ["basic", "wide", "pinch", "scissor"]
MESSAGE_FILTER_SLOP = 0.1        
OVERLAY_ALPHA = 0.5              
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
MAX_LIMB_LENGTH_M = 0.10 # Changed from 0.08

os.makedirs(DATA_DIR, exist_ok=True)

# --- Visualization Constants ---
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]
LIMB_COLORS = [ # Colors for drawing hand connections
    [255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],
    [0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],[0,0,255],[85,0,255],
    [170,0,255],[255,0,255],[255,0,170],[255,0,85],[85,85,85],[170,170,170]
]
VALID_POINT_COLOR = (0, 255, 0)
INVALID_POINT_COLOR = (0, 0, 255)

# --- Data Classes ---
@dataclass
class PendingCaptureMetadata:
    timestamp: Optional[str] = None
    rgb_source_topic: str = ""
    median_filter_applied: bool = False
    openpose_conf_threshold_value: float = OPENPOSE_CONFIDENCE_THRESHOLD
    num_2d_peaks_detected_raw: int = 0
    num_2d_peaks_above_conf: int = 0
    num_3d_points_initial: int = 0

@dataclass
class CaptureData:
    keypoints_rel_3d: Optional[np.ndarray] = None       
    peaks_2d_filtered: Optional[np.ndarray] = None      
    validity_mask_final: Optional[List[bool]] = None    
    capture_metadata: PendingCaptureMetadata = field(default_factory=PendingCaptureMetadata)

@dataclass
class GuiTask:
    command: str
    payload: Any = None

class DepthHandCollector:
    def __init__(self):
        rospy.init_node("hand_collector_node", anonymous=True)
        self.bridge = CvBridge()
        self.is_shutting_down = False
        self._after_id_gui_pump: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("Hand Pose Data Collector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self._gui_queue: queue.Queue[GuiTask] = queue.Queue()
        self._status_var = tk.StringVar(value="Initializing...")
        self._apply_median_filter_var = tk.BooleanVar(value=True)
        self._show_overlay_var = tk.BooleanVar(value=True)
        self._label_counts_vars: Dict[str, tk.StringVar] = {}
        self._rgb_topic_selection_var = tk.StringVar()
        
        self._fx: Optional[float] = None; self._fy: Optional[float] = None
        self._cx: Optional[float] = None; self._cy: Optional[float] = None
        self._camera_info_received = False
        
        self._set_default_rgb_topic()
        self._initialize_openpose_estimator()
        self._initialize_data_storage()
        self._initialize_csv_file()
        self._load_existing_label_counts()
        self._setup_gui()
        self._initialize_ros_subscriptions()
        self._bind_keyboard_shortcuts()
        
        self._waiting_for_label_input = False
        self._process_gui_queue() 

        threading.Thread(target=self.ros_spin_thread, daemon=True).start()
        rospy.loginfo("DepthHandCollector fully initialized.")

    def _set_default_rgb_topic(self):
        self.current_rgb_topic_to_subscribe = RGB_TOPIC_OPTIONS.get(
            "Rectified Color (`image_rect_color`)", 
            list(RGB_TOPIC_OPTIONS.values())[0]
        )
        default_key = next(
            (key for key, value in RGB_TOPIC_OPTIONS.items() if value == self.current_rgb_topic_to_subscribe),
            list(RGB_TOPIC_OPTIONS.keys())[0]
        )
        self._rgb_topic_selection_var.set(default_key)

    def _initialize_openpose_estimator(self):
        try:
            model_path = os.path.join(OPENPOSE_MODEL_DIR, "hand_pose_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"OpenPose hand model not found: {model_path}")
            self._hand_estimator = Hand(model_path)
            self._openpose_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            rospy.loginfo(f"OpenPose using device: {self._openpose_device}")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize OpenPose: {e}")
            if hasattr(self, 'root') and self.root.winfo_exists():
                messagebox.showerror("Fatal Error", f"Failed to initialize OpenPose: {e}", parent=self.root)
            sys.exit(1)

    def _initialize_data_storage(self):
        self._latest_synced_data: Dict[str, Any] = {"rgb": None, "depth": None}
        self._data_access_lock = threading.Lock()
        self._pending_capture = CaptureData()

    def _load_existing_label_counts(self):
        self._label_counts: Dict[str, int] = {label: 0 for label in POSE_LABELS}
        if os.path.exists(CSV_FULL_PATH) and os.path.getsize(CSV_FULL_PATH) > 0:
            try:
                with open(CSV_FULL_PATH, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    if 'label' in reader.fieldnames:
                        for row in reader:
                            label = row.get('label')
                            if label in self._label_counts:
                                self._label_counts[label] += 1
                        rospy.loginfo(f"Loaded label counts from CSV: {self._label_counts}")
            except Exception as e:
                rospy.logerr(f"Error reading CSV for counts: {e}.")
        
        for label_text in POSE_LABELS: 
            self._label_counts_vars[label_text] = tk.StringVar(value=f"{label_text}: {self._label_counts.get(label_text, 0)}")

    def _initialize_csv_file(self):
        try:
            file_exists = os.path.exists(CSV_FULL_PATH)
            with open(CSV_FULL_PATH, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists or os.path.getsize(CSV_FULL_PATH) == 0:
                    rospy.loginfo(f"Creating new CSV file: {CSV_FULL_PATH}")
                    header = ['label', 'timestamp', 'rgb_source_topic',  'calibration_used','median_filter_applied', 
                              'openpose_conf_threshold', 'num_2d_peaks_detected_raw', 
                              'num_2d_peaks_above_conf', 'num_3d_points_initial', 'num_3d_points_final']
                    header.extend([f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')])
                    writer.writerow(header)
                else:
                    rospy.loginfo(f"Appending to existing CSV: {CSV_FULL_PATH}")
        except IOError as e:
            rospy.logfatal(f"Cannot access/write CSV file '{CSV_FULL_PATH}'. Error: {e}")
            self._update_status(f"FATAL ERROR: CSV access: {e}")

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        vis_frame = ttk.Frame(top_frame)
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        left_col = ttk.Frame(vis_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._main_view_label = ttk.Label(left_col) 
        self._main_view_label.pack(fill=tk.BOTH, expand=True)
        self._depth_label = ttk.Label(left_col) 
        self._depth_label.pack(fill=tk.BOTH, expand=True)

        right_col = ttk.Frame(vis_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._pose_2d_label = ttk.Label(right_col) 
        self._pose_2d_label.pack(fill=tk.BOTH, expand=True)
        
        try:
            self._fig_3d = plt.figure(figsize=(4, 3)) 
            self._ax_3d = self._fig_3d.add_subplot(111, projection='3d')
            self._canvas_3d = FigureCanvasTkAgg(self._fig_3d, master=right_col)
            self._canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
             rospy.logerr(f"Failed to initialize 3D plot area: {e}")
             ttk.Label(right_col, text="Error initializing 3D plot.").pack(pady=1, fill=tk.BOTH, expand=True)

        controls_and_counts_frame = ttk.Frame(top_frame)
        controls_and_counts_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        controls_frame = ttk.Labelframe(controls_and_counts_frame, text="Controls", padding=5)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(controls_frame, text="RGB Source:").pack(anchor=tk.W)
        self._rgb_topic_dropdown = ttk.Combobox(controls_frame, textvariable=self._rgb_topic_selection_var, 
                                                values=list(RGB_TOPIC_OPTIONS.keys()), state="readonly", width=35)
        self._rgb_topic_dropdown.pack(fill=tk.X, pady=2)
        self._rgb_topic_dropdown.bind("<<ComboboxSelected>>", self._on_rgb_topic_change)

        self._overlay_check = ttk.Checkbutton(controls_frame, text="Show Depth Overlay", variable=self._show_overlay_var)
        self._overlay_check.pack(anchor=tk.W)
        self._filter_check = ttk.Checkbutton(controls_frame, text="Apply Median Filter (Depth)", variable=self._apply_median_filter_var)
        self._filter_check.pack(anchor=tk.W)
        
        self._capture_btn = ttk.Button(controls_frame, text="Capture Pose (W)", command=self._trigger_capture_async)
        self._capture_btn.pack(fill=tk.X, pady=5)

        counts_frame = ttk.Labelframe(controls_and_counts_frame, text="Label Counts", padding=5)
        counts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        for label_text in POSE_LABELS: 
            ttk.Label(counts_frame, textvariable=self._label_counts_vars[label_text]).pack(anchor=tk.W)

        status_bar = ttk.Label(main_frame, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self._clear_previews() 

    def _clear_previews(self):
        self._gui_queue.put(GuiTask("clear_display", (self._main_view_label, "Waiting for RGB feed...")))
        self._gui_queue.put(GuiTask("clear_display", (self._depth_label, "Waiting for Depth feed...")))
        self._gui_queue.put(GuiTask("clear_display", (self._pose_2d_label, "Press 'W' to capture.")))
        self._gui_queue.put(GuiTask("clear_3d_plot", "Initializing 3D plot..."))

    def _bind_keyboard_shortcuts(self):
        for i, label_text in enumerate(POSE_LABELS):
            self.root.bind(f"<KeyPress-{i+1}>", lambda e, idx=i: self._handle_label_key_press(idx))
        self.root.bind("<KeyPress-q>", self._handle_cancel_key_press)
        self.root.bind("<KeyPress-Q>", self._handle_cancel_key_press)
        self.root.bind("<KeyPress-w>", self._trigger_capture_async) 
        self.root.bind("<KeyPress-W>", self._trigger_capture_async)
    
    def _initialize_ros_subscriptions(self):
        self._unregister_ros_subscribers() 
        try:
            rospy.loginfo(f"Initializing ROS subscribers. RGB Topic: {self.current_rgb_topic_to_subscribe}")
            self.rgb_subscriber_filter = message_filters.Subscriber(self.current_rgb_topic_to_subscribe, Image)
            self.depth_subscriber_filter = message_filters.Subscriber(REGISTERED_DEPTH_TOPIC, Image)
            self.info_subscriber_filter = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)
            
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_subscriber_filter, self.depth_subscriber_filter, self.info_subscriber_filter], 
                10, MESSAGE_FILTER_SLOP
            )
            self.ts.registerCallback(self._ros_synchronized_callback)
            rospy.loginfo("ROS subscribers synchronized.")
            self._update_status("Waiting for synchronized ROS data...")
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)) 
        except Exception as e:
            rospy.logfatal(f"Failed to initialize ROS subscribers: {e}")
            self._update_status("FATAL: Error setting up ROS Sync! Check topics.")
            self._gui_queue.put(GuiTask("set_controls_state", tk.DISABLED))

    def _unregister_ros_subscribers(self):
        if hasattr(self, 'ts') and self.ts is not None:
            self.ts.callbacks.clear()

        for sub_attr_name in ['rgb_subscriber_filter', 'depth_subscriber_filter', 'info_subscriber_filter']:
            if hasattr(self, sub_attr_name):
                subscriber_instance = getattr(self, sub_attr_name)
                if subscriber_instance is not None:
                    if hasattr(subscriber_instance, 'sub') and hasattr(subscriber_instance.sub, 'unregister'):
                         subscriber_instance.sub.unregister()
                    elif hasattr(subscriber_instance, 'unregister'): 
                         subscriber_instance.unregister()
                    setattr(self, sub_attr_name, None) 

        if hasattr(self, 'ts'): 
            self.ts = None
        rospy.loginfo("Unregistered existing ROS subscribers.")

    def _on_rgb_topic_change(self, event=None): 
        new_topic_path = RGB_TOPIC_OPTIONS.get(self._rgb_topic_selection_var.get())
        if new_topic_path and new_topic_path != self.current_rgb_topic_to_subscribe:
            self.current_rgb_topic_to_subscribe = new_topic_path
            self._update_status("Changing RGB topic. Restarting subscribers...")
            self._gui_queue.put(GuiTask("set_controls_state", tk.DISABLED))
            self.root.update_idletasks() 
            self._initialize_ros_subscriptions() 
        elif not new_topic_path:
             messagebox.showwarning("Topic Error", "Invalid RGB topic selected.", parent=self.root)

    def _ros_synchronized_callback(self, rgb_msg, depth_msg, info_msg):
        if self.is_shutting_down: return
        
        if not self._camera_info_received: 
            if hasattr(info_msg, 'K') and len(info_msg.K) == 9 and info_msg.K[0] > 0: 
                self._fx, self._fy = info_msg.K[0], info_msg.K[4]
                self._cx, self._cy = info_msg.K[2], info_msg.K[5]
                self._camera_info_received = True
                rospy.loginfo_once(f"Camera intrinsics received from topic: Fx={self._fx:.2f}, Fy={self._fy:.2f}, Cx={self._cx:.2f}, Cy={self._cy:.2f}")
                self._update_status("Camera info received. Ready for capture.")
            else:
                rospy.logwarn_throttle(10, "Received CameraInfo message but K matrix is invalid or not populated yet.")
                return 
        
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image_16uc1 = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            with self._data_access_lock:
                self._latest_synced_data["rgb"] = rgb_image
                self._latest_synced_data["depth"] = depth_image_16uc1
            self._gui_queue.put(GuiTask("update_previews", (rgb_image.copy(), depth_image_16uc1.copy())))
        except CvBridgeError as e:
            rospy.logerr_throttle(5, f"CvBridge Error in synchronized_callback: {e}")
        except Exception as e_sync:
             rospy.logerr_throttle(5, f"Error processing synchronized messages: {e_sync}")

    def _trigger_capture_async(self, event=None): 
        if self._waiting_for_label_input:
            rospy.logwarn("Capture attempt ignored: already waiting for a label.")
            return
        self._gui_queue.put(GuiTask("set_controls_state", tk.DISABLED))
        threading.Thread(target=self._process_capture, daemon=True).start()

    def _filter_3d_by_limb_length(self, keypoints_3d_rel, validity_mask):
        points_filtered = keypoints_3d_rel.copy()
        new_validity_mask = list(validity_mask) 

        MAX_ITERATIONS = 5 # Safety break for the while loop
        for iteration in range(MAX_ITERATIONS):
            num_removed_in_pass = 0
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                # Ensure indices are within bounds of the current mask
                if not (0 <= p1_idx < len(new_validity_mask) and 0 <= p2_idx < len(new_validity_mask)):
                    continue

                if new_validity_mask[p1_idx] and new_validity_mask[p2_idx]:
                    
                    p1 = points_filtered[p1_idx]
                    p2 = points_filtered[p2_idx]
                    
                    # Check if points are still valid (not NaNs introduced by previous removals in the same pass)
                    if np.isnan(p1).any() or np.isnan(p2).any():
                        continue

                    dist_sq = np.sum((p1 - p2)**2)
                    
                    if dist_sq > MAX_LIMB_LENGTH_M**2:
                        # Anomaly detected: this limb is too long.
                        # Invalidate the point that is further from the origin (assumed to be the wrist
                        # if coordinates are relative, or camera origin otherwise).
                        dist_p1_sq_from_origin = np.sum(p1**2)
                        dist_p2_sq_from_origin = np.sum(p2**2)
                        
                        if dist_p1_sq_from_origin > dist_p2_sq_from_origin:
                            if new_validity_mask[p1_idx]: # Only act if it's currently considered valid
                                points_filtered[p1_idx] = np.nan
                                new_validity_mask[p1_idx] = False
                                num_removed_in_pass += 1
                        else: # p2 is further or they are equidistant (invalidate p2 by default)
                            if new_validity_mask[p2_idx]: # Only act if it's currently considered valid
                                points_filtered[p2_idx] = np.nan
                                new_validity_mask[p2_idx] = False
                                num_removed_in_pass += 1
            
            if num_removed_in_pass == 0:
                break # No changes in this pass, stable state reached
        
        if iteration == MAX_ITERATIONS - 1 and num_removed_in_pass > 0:
            rospy.logwarn_throttle(10, f"Limb length filter reached max iterations ({MAX_ITERATIONS}) and still making changes.")
                        
        return points_filtered, new_validity_mask

    def _process_capture(self):
        with self._data_access_lock:
            rgb_image_cap = self._latest_synced_data.get("rgb")
            depth_image_cap = self._latest_synced_data.get("depth")

        if rgb_image_cap is None or depth_image_cap is None:
            self._update_status("Error: Missing image data for capture!"); self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return
        if not self._camera_info_received or self._fx is None:
            self._update_status("Error: Camera intrinsics not yet received!"); self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        self._update_status("Detecting hand...")
        current_metadata = PendingCaptureMetadata(timestamp=datetime.datetime.now().isoformat(),
                                                  rgb_source_topic=self.current_rgb_topic_to_subscribe,
                                                  median_filter_applied=self._apply_median_filter_var.get())
        
        depth_image_processed = cv2.medianBlur(depth_image_cap, MEDIAN_FILTER_KERNEL_SIZE) if current_metadata.median_filter_applied else depth_image_cap
        
        try:
            all_peaks_2d_raw = self._hand_estimator(rgb_image_cap) 
            if all_peaks_2d_raw is None or not isinstance(all_peaks_2d_raw, np.ndarray) or all_peaks_2d_raw.ndim != 2:
                raise ValueError("Invalid OpenPose output or hand not detected")
        except Exception as e_pose:
            self._update_status(f"Hand Detection Error: {e_pose}"); self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        current_metadata.num_2d_peaks_detected_raw = all_peaks_2d_raw.shape[0]
        
        peaks_2d_for_capture_data = np.full((21,2), np.nan, dtype=np.float32)
        has_confidence = all_peaks_2d_raw.shape[1] >= 3
        current_metadata.openpose_conf_threshold_value = OPENPOSE_CONFIDENCE_THRESHOLD if has_confidence else -1.0
        
        num_passed_conf = 0
        for i in range(min(21, current_metadata.num_2d_peaks_detected_raw)):
            x_coord, y_coord = all_peaks_2d_raw[i, 0], all_peaks_2d_raw[i, 1]
            passes_check = False
            if has_confidence:
                if all_peaks_2d_raw[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD: 
                    passes_check = True
            else: 
                passes_check = True 
            
            if passes_check:
                peaks_2d_for_capture_data[i] = [x_coord, y_coord]
                num_passed_conf += 1
        
        current_metadata.num_2d_peaks_above_conf = num_passed_conf
        
        if current_metadata.num_2d_peaks_above_conf == 0:
            status_msg = f"No 2D keypoints passed criteria. Raw: {current_metadata.num_2d_peaks_detected_raw}."
            self._update_status(status_msg); self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return
        
        keypoints_3d_camera_frame_full = np.full((21, 3), np.nan, dtype=np.float32)
        num_projected_3d = 0
        for i in range(21): 
            if not np.isnan(peaks_2d_for_capture_data[i, 0]): 
                u, v = int(round(peaks_2d_for_capture_data[i, 0])), int(round(peaks_2d_for_capture_data[i, 1]))
                z_meters = self._get_depth_from_neighborhood(depth_image_processed, u, v)
                if not np.isnan(z_meters):
                    x_cam = (u - self._cx) * z_meters / self._fx
                    y_cam = (v - self._cy) * z_meters / self._fy
                    keypoints_3d_camera_frame_full[i] = [x_cam, y_cam, z_meters]
                    num_projected_3d +=1
        
        current_metadata.num_3d_points_initial = num_projected_3d
        
        keypoints_3d_relative = np.full_like(keypoints_3d_camera_frame_full, np.nan)
        if not np.isnan(keypoints_3d_camera_frame_full[0]).any(): 
            wrist_3d_coords = keypoints_3d_camera_frame_full[0].copy()
            keypoints_3d_relative = keypoints_3d_camera_frame_full - wrist_3d_coords
            keypoints_3d_relative[0] = [0.0, 0.0, 0.0] 
        else: 
            keypoints_3d_relative = keypoints_3d_camera_frame_full 

        keypoints_3d_relative_filtered, final_validity_mask = self._filter_3d_outliers(keypoints_3d_relative)
        keypoints_3d_relative_filtered, final_validity_mask = self._filter_3d_by_limb_length(
            keypoints_3d_relative_filtered, final_validity_mask
        )

        num_final_valid_3d_points = np.sum(final_validity_mask)
        
        self._pending_capture = CaptureData(
            keypoints_rel_3d=keypoints_3d_relative_filtered,
            peaks_2d_filtered=peaks_2d_for_capture_data, 
            validity_mask_final=final_validity_mask,
            capture_metadata=current_metadata
        )

        self._gui_queue.put(GuiTask("update_pose_preview", (rgb_image_cap.copy(), self._pending_capture)))
        self._gui_queue.put(GuiTask("update_3d_plot", (keypoints_3d_relative_filtered, f"Relative 3D ({num_final_valid_3d_points} valid)")))

        if num_final_valid_3d_points < MIN_VALID_KEYPOINTS_FOR_SAVE or not final_validity_mask[0]:
            status_msg = f"Capture failed: {num_final_valid_3d_points} points (min {MIN_VALID_KEYPOINTS_FOR_SAVE}). Wrist valid: {final_validity_mask[0]}."
            self._update_status(status_msg); self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        self._waiting_for_label_input = True
        label_prompt = ", ".join([f"{i+1}={L}" for i, L in enumerate(POSE_LABELS)])
        self._update_status(f"Pose captured ({num_final_valid_3d_points} pts). Label: [{label_prompt}] or Q=Cancel.")

    def _handle_label_key_press(self, label_idx: int):
        if not self._waiting_for_label_input or not (0 <= label_idx < len(POSE_LABELS)): return
        selected_label = POSE_LABELS[label_idx]
        rospy.loginfo(f"Label '{selected_label}' selected. Saving pose...")
        
        self._save_pending_capture_to_csv(selected_label)
        
        self._label_counts[selected_label] = self._label_counts.get(selected_label, 0) + 1
        self._gui_queue.put(GuiTask("update_label_counts")) 

        self._waiting_for_label_input = False
        self._pending_capture = CaptureData() 
        self._update_status("Pose saved. Ready for next capture (Press W).")
        self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL))

    def _handle_cancel_key_press(self, event=None): 
        if not self._waiting_for_label_input: return
        rospy.loginfo("Cancel key pressed. Discarding current capture.")
        self._waiting_for_label_input = False
        self._pending_capture = CaptureData() 
        self._update_status("Capture cancelled. Ready for next capture (Press W).")
        self._gui_queue.put(GuiTask("clear_3d_plot", "Capture Cancelled")) 
        self._gui_queue.put(GuiTask("clear_display", (self._pose_2d_label, "Capture Cancelled")))
        self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL))

    def _save_pending_capture_to_csv(self, assigned_label: str):
        data_to_save = self._pending_capture
        points_to_save = data_to_save.keypoints_rel_3d
        metadata = data_to_save.capture_metadata
        validity_mask = data_to_save.validity_mask_final

        if points_to_save is None or metadata is None or validity_mask is None:
            rospy.logerr("Attempted to save, but pending data is incomplete!"); return

        num_valid_to_save = np.sum(validity_mask)
        self._update_status(f"Saving {num_valid_to_save} valid points as '{assigned_label}'...")

        try:
            with open(CSV_FULL_PATH, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                row = [assigned_label, metadata.timestamp, metadata.rgb_source_topic, False,
                        metadata.median_filter_applied, metadata.openpose_conf_threshold_value,
                        metadata.num_2d_peaks_detected_raw, metadata.num_2d_peaks_above_conf,
                        metadata.num_3d_points_initial, num_valid_to_save]
                coords_flat = []
                for i in range(21): 
                    if validity_mask[i] and not np.isnan(points_to_save[i]).any():
                        coords_flat.extend([f"{c:.6f}" for c in points_to_save[i]])
                    else:
                        coords_flat.extend(["", "", ""]) 
                row.extend(coords_flat)
                writer.writerow(row)
            rospy.loginfo(f"Pose '{assigned_label}' data appended to {CSV_FULL_PATH}")
        except IOError as e:
            rospy.logerr(f"IOError writing to CSV: {e}")
            self._update_status(f"Error writing CSV: {e}")
        except Exception as e_save:
            rospy.logerr(f"Unexpected error saving CSV: {e_save}")
            self._update_status("Error saving CSV! Check logs.")

    def on_close(self):
        if self.is_shutting_down: return
        rospy.loginfo("Close button pressed. Initiating shutdown...")
        self.is_shutting_down = True
        if self._after_id_gui_pump:
            try: self.root.after_cancel(self._after_id_gui_pump)
            except tk.TclError: pass 
        self._unregister_ros_subscribers()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("GUI closed by user")
        try:
            if hasattr(self, '_fig_3d') and self._fig_3d.canvas.get_tk_widget().winfo_exists() and plt.fignum_exists(self._fig_3d.number):
                plt.close(self._fig_3d)
        except Exception as e_plt: rospy.logwarn(f"Error closing matplotlib figure: {e_plt}")
        
        if hasattr(self, 'root') and self.root.winfo_exists():
            try:
                self.root.destroy() 
            except tk.TclError: pass 
        rospy.loginfo("GUI resources released. Collector shut down.")

    def _process_gui_queue(self):
        if self.is_shutting_down: return
        try:
            while not self._gui_queue.empty():
                task = self._gui_queue.get_nowait()
                if not (hasattr(self, 'root') and self.root.winfo_exists()): break 

                if task.command == "update_status": self._status_var.set(str(task.payload))
                elif task.command == "update_previews": self._update_gui_previews(*task.payload)
                elif task.command == "update_pose_preview": self._update_gui_2d_pose_display(*task.payload)
                elif task.command == "update_3d_plot": self._update_gui_3d_plot(*task.payload)
                elif task.command == "clear_display": self._clear_gui_display(*task.payload)
                elif task.command == "clear_3d_plot": self._clear_gui_3d_plot(task.payload)
                elif task.command == "set_controls_state": self._set_gui_controls_state_all(task.payload)
                elif task.command == "update_label_counts": self._update_gui_label_counts()
                else: rospy.logwarn(f"Unknown GUI command: {task.command}")
                self._gui_queue.task_done()
        except queue.Empty: pass
        except Exception as e_pump: rospy.logerr(f"Error processing GUI queue: {e_pump}")
        finally:
            if not self.is_shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
                self._after_id_gui_pump = self.root.after(50, self._process_gui_queue) 

    def _update_gui_previews(self, rgb_img, depth_img_16u):
        if self._show_overlay_var.get():
            overlay = self._create_overlay_image(rgb_img, depth_img_16u)
            self._update_gui_single_display(self._main_view_label, overlay if overlay is not None else rgb_img)
        else:
            self._update_gui_single_display(self._main_view_label, rgb_img)
        
        depth_colorized = self._colorize_depth_image(depth_img_16u)
        self._update_gui_single_display(self._depth_label, depth_colorized)

    def _update_gui_2d_pose_display(self, rgb_image_base, capture_data_obj: CaptureData):
        display_img = rgb_image_base.copy()
        if capture_data_obj.peaks_2d_filtered is not None and capture_data_obj.validity_mask_final is not None:
            # Draw connections
            for conn_idx, (p1_idx, p2_idx) in enumerate(HAND_CONNECTIONS):
                # Check if both points for the connection are valid (not NaN and passed final 3D validity)
                # Use peaks_2d_filtered to get coordinates for drawing, but validity_mask_final to decide if connection should be drawn
                if not np.isnan(capture_data_obj.peaks_2d_filtered[p1_idx,0]) and \
                   not np.isnan(capture_data_obj.peaks_2d_filtered[p2_idx,0]) and \
                   capture_data_obj.validity_mask_final[p1_idx] and \
                   capture_data_obj.validity_mask_final[p2_idx]:
                    
                    pt1 = tuple(capture_data_obj.peaks_2d_filtered[p1_idx].astype(int))
                    pt2 = tuple(capture_data_obj.peaks_2d_filtered[p2_idx].astype(int))
                    cv2.line(display_img, pt1, pt2, LIMB_COLORS[conn_idx % len(LIMB_COLORS)], 2)
            # Draw keypoints
            for i in range(21): 
                if not np.isnan(capture_data_obj.peaks_2d_filtered[i, 0]): # If 2D peak exists
                    u, v = int(capture_data_obj.peaks_2d_filtered[i, 0]), int(capture_data_obj.peaks_2d_filtered[i, 1])
                    # Color based on final 3D validity
                    color = VALID_POINT_COLOR if capture_data_obj.validity_mask_final[i] else INVALID_POINT_COLOR
                    cv2.circle(display_img, (u, v), 3, color, -1) 
        self._update_gui_single_display(self._pose_2d_label, display_img)

    def _update_gui_3d_plot(self, points_3d_relative_coords, title_text):
        if not (hasattr(self, '_ax_3d') and self._canvas_3d.get_tk_widget().winfo_exists()): return
        self._ax_3d.clear()
        
        if points_3d_relative_coords is not None and np.sum(~np.isnan(points_3d_relative_coords)) > 0:
            valid_indices = ~np.isnan(points_3d_relative_coords).any(axis=1)
            valid_points = points_3d_relative_coords[valid_indices]

            if valid_points.shape[0] > 0:
                x = valid_points[:, 0]
                y = -valid_points[:, 1]
                z = -valid_points[:, 2]
                
                self._ax_3d.scatter(x, y, z, c='r', marker='o')

                for conn_idx, (p1_idx, p2_idx) in enumerate(HAND_CONNECTIONS):
                    if valid_indices[p1_idx] and valid_indices[p2_idx]:
                        p1 = points_3d_relative_coords[p1_idx]
                        p2 = points_3d_relative_coords[p2_idx]
                        self._ax_3d.plot([p1[0], p2[0]], [-p1[1], -p2[1]], [-p1[2], -p2[2]], color=[c/255.0 for c in LIMB_COLORS[conn_idx % len(LIMB_COLORS)]])

                all_vals = np.array([x, y, z]).flatten()
                min_vals = np.nanmin(valid_points, axis=0)
                max_vals = np.nanmax(valid_points, axis=0)
                centers = (min_vals + max_vals) / 2.0
                ranges = (max_vals - min_vals).max() / 2.0
                if ranges < 0.05: ranges = 0.05

                self._ax_3d.set_xlim(centers[0] - ranges, centers[0] + ranges)
                self._ax_3d.set_ylim(-(centers[1] + ranges), -(centers[1] - ranges))
                self._ax_3d.set_zlim(-(centers[2] + ranges), -(centers[2] - ranges))
        else:
            self._ax_3d.set_xlim([-0.15, 0.15])
            self._ax_3d.set_ylim([-0.15, 0.15])
            self._ax_3d.set_zlim([-0.15, 0.15])

        self._ax_3d.set_xlabel('X (m)')
        self._ax_3d.set_ylabel('-Y (m)')
        self._ax_3d.set_zlabel('-Z (m)')
        self._ax_3d.set_title(title_text)

        try:
            self._ax_3d.set_aspect('equal', adjustable='box')
        except NotImplementedError:
            self._ax_3d.set_aspect('auto')
        
        self._canvas_3d.draw_idle()

    def _clear_gui_display(self, target_label_widget, text_to_show):
        blank_placeholder = np.zeros((DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank_placeholder, text_to_show, (10, DISPLAY_MAX_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._update_gui_single_display(target_label_widget, blank_placeholder)
        
    def _clear_gui_3d_plot(self, title_text):
        if not (hasattr(self, '_ax_3d') and self._canvas_3d.get_tk_widget().winfo_exists()): return
        self._ax_3d.clear()
        self._ax_3d.set_title(title_text)
        self._ax_3d.set_xlim([-0.3, 0.3]); self._ax_3d.set_ylim([-0.3, 0.3]); self._ax_3d.set_zlim([-0.3, 0.3])
        self._canvas_3d.draw_idle()

    def _set_gui_controls_state_all(self, target_tk_state: str): 
        widgets_to_toggle = [self._capture_btn, self._filter_check, self._overlay_check, self._rgb_topic_dropdown]
        for widget in widgets_to_toggle:
            if hasattr(widget, 'configure'): widget.configure(state=target_tk_state)
            
    def _update_gui_label_counts(self):
        for label_text, count_var in self._label_counts_vars.items(): 
            count_var.set(f"{label_text}: {self._label_counts.get(label_text, 0)}")

    def _update_gui_single_display(self, tk_label_widget, cv_image_bgr):
        if cv_image_bgr is None or not (hasattr(tk_label_widget, 'winfo_exists') and tk_label_widget.winfo_exists()): return
        try:
            img_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT), interpolation=cv2.INTER_NEAREST)
            img_pil = PILImage.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            tk_label_widget.configure(image=img_tk)
            tk_label_widget.image = img_tk 
        except Exception as e_disp: rospy.logwarn_throttle(5, f"Error updating GUI display: {e_disp}")

    def _colorize_depth_image(self, depth_16uc1_image):
        min_d, max_d = VALID_DEPTH_THRESHOLD_MM
        depth_clipped = np.clip(depth_16uc1_image, min_d, max_d)
        if np.max(depth_clipped) - np.min(depth_clipped) > 0: 
            norm_image = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            norm_image = np.zeros_like(depth_clipped, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)

    def _create_overlay_image(self, rgb_bgr_image, depth_16uc1_image):
        if rgb_bgr_image is None or depth_16uc1_image is None: return rgb_bgr_image 
        
        colorized_depth = self._colorize_depth_image(depth_16uc1_image)
        if rgb_bgr_image.shape[:2] != colorized_depth.shape[:2]:
            colorized_depth = cv2.resize(colorized_depth, (rgb_bgr_image.shape[1], rgb_bgr_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        return cv2.addWeighted(rgb_bgr_image, OVERLAY_ALPHA, colorized_depth, 1 - OVERLAY_ALPHA, 0)
    
    def _get_depth_from_neighborhood(self, depth_map_mm, u_px, v_px):
        if not (0 <= v_px < depth_map_mm.shape[0] and 0 <= u_px < depth_map_mm.shape[1]): return np.nan
        
        radius = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_start, y_end = max(0, v_px - radius), min(depth_map_mm.shape[0], v_px + radius + 1)
        x_start, x_end = max(0, u_px - radius), min(depth_map_mm.shape[1], u_px + radius + 1)
        
        neighborhood = depth_map_mm[y_start:y_end, x_start:x_end]
        valid_depths = neighborhood[(neighborhood >= VALID_DEPTH_THRESHOLD_MM[0]) & (neighborhood <= VALID_DEPTH_THRESHOLD_MM[1])]
        
        if valid_depths.size < max(1, (DEPTH_NEIGHBORHOOD_SIZE**2)//4): return np.nan
        if np.std(valid_depths) > DEPTH_STD_DEV_THRESHOLD_MM: return np.nan 
        
        return float(np.median(valid_depths) / 1000.0) 

    def _filter_3d_outliers(self, keypoints_3d_relative_to_wrist):
        filtered_points = keypoints_3d_relative_to_wrist.copy()
        current_validity_mask = ~np.isnan(filtered_points).any(axis=1)

        if not current_validity_mask[0]: 
            return filtered_points, current_validity_mask.tolist()
        
        max_dist_squared_from_wrist = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21): 
            if current_validity_mask[i]:
                dist_sq = np.sum(filtered_points[i]**2) 
                if dist_sq > max_dist_squared_from_wrist:
                    filtered_points[i] = np.nan 
                    current_validity_mask[i] = False
        return filtered_points, current_validity_mask.tolist()
    
    def _update_status(self, message: str):
        self._gui_queue.put(GuiTask("update_status", message))
        
    def ros_spin_thread(self):
        try:
            rospy.spin()
            rospy.loginfo("ROS spin thread finished.")
        except rospy.ROSInterruptException:
            rospy.loginfo("ROS spin thread interrupted (ROS shutdown).")
        except Exception as e_ros_spin:
            rospy.logerr(f"Error in ROS spin thread: {e_ros_spin}")
        
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e_mainloop:
            rospy.logerr(f"Exception in Tkinter mainloop: {e_mainloop}")
        finally:
            if not self.is_shutting_down:
                self.on_close()

if __name__ == "__main__":
    collector_instance = None
    try:
        collector_instance = DepthHandCollector()
        collector_instance.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted by ROS master (e.g., Ctrl+C in terminal).")
    except KeyboardInterrupt: 
        rospy.loginfo("Script interrupted by user (Ctrl+C).")
    except SystemExit:
        rospy.logwarn("SystemExit called, shutting down.")
    except Exception as e_main:
        rospy.logfatal(f"Unhandled exception in main execution block: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if collector_instance and hasattr(collector_instance, 'is_shutting_down') and not collector_instance.is_shutting_down:
            rospy.loginfo("Performing final cleanup from main's finally block...")
            collector_instance.on_close() 
        if not rospy.is_shutdown():
            rospy.loginfo("ROS is not shutdown in main finally block, signaling shutdown now.")
            rospy.signal_shutdown("Script main block finished or unhandled error.")
        rospy.loginfo("Data collection script has finished execution.")
        print("\nExiting DepthHandCollector script.")