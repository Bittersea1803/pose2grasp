import os
import sys
import time 
import queue
import threading
import csv
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
# import yaml
import datetime

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
from PIL import Image as PILImage
from PIL import ImageTk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Path Configuration ---
def get_project_root():
    """Gets the absolute path to the project's root directory (e.g., 'pose2grasp/')."""
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
    from src import util as openpose_util
    rospy.loginfo("Successfully imported OpenPose modules from src package.")

except ImportError as e_pkg:
    rospy.logwarn(f"Could not import OpenPose from src package (Error: {e_pkg}). Trying alternative import by adding 'src' to path...")
    try:
        openpose_src_path = os.path.join(OPENPOSE_PYTHON_PATH, "src")
        if not os.path.isdir(openpose_src_path):
             raise ImportError(f"OpenPose 'src' directory not found at: {openpose_src_path}")
        sys.path.insert(0, openpose_src_path)
        from hand import Hand
        import util as openpose_util
        rospy.loginfo("Successfully imported OpenPose modules by adding 'src' dir to path.")
    except ImportError as e_src:
        print(f"Fatal Error: Cannot import OpenPose from {OPENPOSE_PYTHON_PATH} or its src subdirectory.")
        print(f"ImportError (package attempt): {e_pkg}")
        print(f"ImportError (src dir attempt): {e_src}")
        sys.exit(1)


# --- Configuration Constants ---
RGB_TOPIC_RAW = "/camera/rgb/image_color"
RGB_TOPIC_RECT_COLOR = "/camera/rgb/image_rect_color"
RGB_SOURCE_SELECTION_DEFAULT = "rect_color"
RGB_TOPIC_OPTIONS = {
    "Rectified Color (`image_rect_color`)": RGB_TOPIC_RECT_COLOR,
    "Raw Color (`image_color`)": RGB_TOPIC_RAW 
}

REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

CSV_FILENAME = "collected_hand_poses.csv" 
CSV_FULL_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

VALID_DEPTH_THRESHOLD_MM = (400, 1500)
MIN_VALID_KEYPOINTS_FOR_SAVE = 16
DISPLAY_MAX_WIDTH = 320
DISPLAY_MAX_HEIGHT = 240
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
OUTLIER_XYZ_THRESHOLD_M = 0.5
MEDIAN_FILTER_KERNEL_SIZE = 3
POSE_LABELS = ["basic", "wide", "pinch", "scissor"]
MESSAGE_FILTER_SLOP = 0.1
OVERLAY_ALPHA = 0.5
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2

os.makedirs(DATA_DIR, exist_ok=True)

# --- Visualization Constants ---
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]
LIMB_COLORS = [
    [255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],
    [0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],[0,0,255],[85,0,255],
    [170,0,255],[255,0,255],[255,0,170],[255,0,85],[85,85,85],[170,170,170]
]
VALID_POINT_COLOR = (0, 255, 0); INVALID_POINT_COLOR = (0, 0, 255)

# --- Data Classes ---
@dataclass
class PendingCaptureMetadata:
    timestamp: Optional[str] = None
    rgb_source_topic: str = ""
    calibration_used: bool = False
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
        rospy.loginfo("ROS Node Initialized: hand_collector_node")
        self.is_shutting_down = False
        self._after_id_gui_pump: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("Hand Pose Data Collector (No External Calibration)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._gui_queue: queue.Queue[GuiTask] = queue.Queue()

        self._status_var = tk.StringVar(value="Initializing...")
        self._apply_median_filter_var = tk.BooleanVar(value=True)
        self._show_overlay_var = tk.BooleanVar(value=True)
        self._label_counts_vars: Dict[str, tk.StringVar] = {}
        self._rgb_topic_selection_var = tk.StringVar()
        
        # Camera Info from Topic (sole source of intrinsics now)
        self._fx_topic: Optional[float] = None; self._fy_topic: Optional[float] = None
        self._cx_topic: Optional[float] = None; self._cy_topic: Optional[float] = None
        self._camera_info_topic_received = False

        self.current_rgb_topic_to_subscribe = RGB_TOPIC_RECT_COLOR
        if RGB_SOURCE_SELECTION_DEFAULT.lower() == "raw":
             self.current_rgb_topic_to_subscribe = RGB_TOPIC_RAW
        
        default_dropdown_key = ""
        for key, value in RGB_TOPIC_OPTIONS.items():
            if value == self.current_rgb_topic_to_subscribe:
                default_dropdown_key = key
                break
        self._rgb_topic_selection_var.set(default_dropdown_key or list(RGB_TOPIC_OPTIONS.keys())[0])

        self._initialize_openpose_estimator()
        self._initialize_data_storage_vars()
        self._initialize_csv_file()
        self._load_existing_label_counts()
        self._setup_gui_layout_widgets()
        self._initialize_ros_topic_subscriptions()

        self._waiting_for_label_input = False

        for i, label_text_key in enumerate(POSE_LABELS):
            key_char_bind = str(i + 1)
            if label_text_key.lower() == "other":
                if "o" not in [str(j+1) for j in range(len(POSE_LABELS)) if j != i]:
                    key_char_bind = "o"
            self.root.bind(f"<KeyPress-{key_char_bind.lower()}>", lambda event, idx=i: self._handle_label_key_press(idx))
            if key_char_bind.isalpha():
                 self.root.bind(f"<KeyPress-{key_char_bind.upper()}>", lambda event, idx=i: self._handle_label_key_press(idx))

        self.root.bind("<KeyPress-q>", self._handle_cancel_key_press)
        self.root.bind("<KeyPress-Q>", self._handle_cancel_key_press)

        self._process_gui_queue()

        if not rospy.is_shutdown():
            threading.Thread(target=self.ros_spin_thread_target, daemon=True).start()
        rospy.loginfo("DepthHandCollector fully initialized (No external calibration).")


    def _initialize_openpose_estimator(self):
        try:
            openpose_model_file = os.path.join(OPENPOSE_MODEL_DIR, "hand_pose_model.pth")
            if not os.path.exists(openpose_model_file):
                raise FileNotFoundError(f"OpenPose hand model not found: {openpose_model_file}")
            self._hand_estimator = Hand(openpose_model_file)
            self._openpose_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            rospy.loginfo(f"OpenPose using device: {self._openpose_device}")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize OpenPose estimator: {e}")
            parent_window = self.root if hasattr(self, 'root') and self.root.winfo_exists() else None
            messagebox.showerror("Fatal Error", f"Failed to initialize OpenPose: {e}", parent=parent_window)
            sys.exit(1)

    def _initialize_data_storage_vars(self):
        self._latest_synced_data: Dict[str, Any] = {"rgb": None, "depth": None, "stamp": None, "info_msg": None}
        self._data_access_lock = threading.Lock() 
        self._pending_capture_data_instance = CaptureData() 

    def _load_existing_label_counts(self):
        self._label_counts: Dict[str, int] = {label: 0 for label in POSE_LABELS}
        if os.path.exists(CSV_FULL_PATH) and os.path.getsize(CSV_FULL_PATH) > 0:
            try:
                with open(CSV_FULL_PATH, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    if 'label' not in reader.fieldnames:
                        rospy.logwarn(
                            f"CSV file '{CSV_FULL_PATH}' does not contain 'label' column in header. "
                            "Cannot load existing counts."
                        )
                    else:
                        rospy.loginfo(f"Loading existing label counts from '{CSV_FULL_PATH}'...")
                        for row in reader:
                            label_from_csv = row.get('label')
                            if label_from_csv in self._label_counts:
                                self._label_counts[label_from_csv] += 1
                        rospy.loginfo(f"Finished loading counts from CSV: {self._label_counts}")
            except Exception as e:
                rospy.logerr(f"Error reading CSV ('{CSV_FULL_PATH}') for populating label counts: {e}. "
                               "Counts will start from 0 for this session.")
        else:
            rospy.loginfo(f"Existing CSV file not found or empty. Label counts will start from 0.")
        
        for label_text in POSE_LABELS: 
            self._label_counts_vars[label_text] = tk.StringVar(value=f"{label_text}: {self._label_counts.get(label_text, 0)}")
        
        if hasattr(self, 'root') and self.root.winfo_exists() and hasattr(self, '_update_label_counts_gui_display'):
            self._gui_queue.put(GuiTask("update_label_counts_display", None))

    def _initialize_csv_file(self):
        try:
            csv_dir = os.path.dirname(CSV_FULL_PATH)
            if not os.path.exists(csv_dir): os.makedirs(csv_dir)
            
            if csv_dir and not os.access(csv_dir, os.W_OK):
                raise IOError(f"No write permission for directory: {csv_dir}")
            if os.path.exists(CSV_FULL_PATH) and not os.access(CSV_FULL_PATH, os.W_OK):
                 raise IOError(f"No write permission for file: {CSV_FULL_PATH}")

            file_exists_and_has_content = os.path.exists(CSV_FULL_PATH) and os.path.getsize(CSV_FULL_PATH) > 0
            with open(CSV_FULL_PATH, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists_and_has_content:
                    rospy.loginfo(f"Creating/initializing CSV file: {CSV_FULL_PATH}")
                    header = ['label', 'timestamp', 'rgb_source_topic', 'calibration_used', # Will always be False
                                'median_filter_applied', 'openpose_conf_threshold',
                                'num_2d_peaks_detected_raw', 'num_2d_peaks_above_conf',
                                'num_3d_points_initial', 'num_3d_points_final']
                    header.extend([f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')])
                    writer.writerow(header)
                else:
                    rospy.loginfo(f"Appending to existing CSV file: {CSV_FULL_PATH}")
        except IOError as e:
            rospy.logfatal(f"Cannot access/write CSV file '{CSV_FULL_PATH}'. Error: {e}")
            self._update_gui_status_message(f"FATAL ERROR: Cannot access CSV: {e}")
            if hasattr(self, '_capture_btn'): self._gui_queue.put(GuiTask("set_controls_state", tk.DISABLED))
        except Exception as e_gen:
            rospy.logfatal(f"Unexpected error initializing CSV: {e_gen}")
            self._update_gui_status_message(f"FATAL ERROR: CSV Init: {e_gen}")


    def _setup_gui_layout_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_section_frame = ttk.Frame(main_frame)
        top_section_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        vis_frame_outer = ttk.Frame(top_section_frame)
        vis_frame_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        vis_frame = ttk.Frame(vis_frame_outer)
        vis_frame.pack(fill=tk.BOTH, expand=True)
        left_vis_col = ttk.Frame(vis_frame)
        left_vis_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._main_view_label = ttk.Label(left_vis_col, text="Waiting for RGB/Overlay...")
        self._main_view_label.pack(pady=1, fill=tk.BOTH, expand=True)
        self._depth_label = ttk.Label(left_vis_col, text="Waiting for Depth...")
        self._depth_label.pack(pady=1, fill=tk.BOTH, expand=True)
        self._clear_depth_preview_display("Waiting for Depth...")
        right_vis_col = ttk.Frame(vis_frame)
        right_vis_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._pose_2d_label = ttk.Label(right_vis_col, text="Waiting for 2D Pose...")
        self._pose_2d_label.pack(pady=1, fill=tk.BOTH, expand=True)
        self._clear_2d_pose_preview_display("Capture to see 2D Pose")
        try:
            self._fig_3d = plt.figure(figsize=(4,3))
            self._ax_3d = self._fig_3d.add_subplot(111, projection='3d')
            self._canvas_3d = FigureCanvasTkAgg(self._fig_3d, master=right_vis_col)
            self._canvas_3d_widget = self._canvas_3d.get_tk_widget()
            self._canvas_3d_widget.pack(pady=1, fill=tk.BOTH, expand=True)
            self._clear_3d_plot_display("Initializing 3D Plot...")
        except Exception as e:
            rospy.logerr(f"Failed to initialize 3D plot area: {e}")
            ttk.Label(right_vis_col, text="Error initializing 3D plot area.").pack(pady=1, fill=tk.BOTH, expand=True)
        right_controls_and_counts_frame = ttk.Frame(top_section_frame)
        right_controls_and_counts_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
        controls_outer_frame = ttk.Labelframe(right_controls_and_counts_frame, text="Controls", padding="5")
        controls_outer_frame.pack(fill=tk.X, pady=(0,5))
        topic_select_frame = ttk.Frame(controls_outer_frame)
        topic_select_frame.pack(fill=tk.X, pady=2)
        ttk.Label(topic_select_frame, text="RGB Source:").pack(side=tk.LEFT, padx=(0,2))
        self._rgb_topic_dropdown = ttk.Combobox(topic_select_frame,
                                                textvariable=self._rgb_topic_selection_var,
                                                values=list(RGB_TOPIC_OPTIONS.keys()),
                                                state="readonly", width=30)
        self._rgb_topic_dropdown.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        self._apply_topic_btn = ttk.Button(topic_select_frame, text="Apply",
                                            command=self._on_apply_rgb_topic_change)
        self._apply_topic_btn.pack(side=tk.LEFT)
        checkbox_frame = ttk.Frame(controls_outer_frame)
        checkbox_frame.pack(fill=tk.X, pady=2)
        self._overlay_checkbutton = ttk.Checkbutton(checkbox_frame, text="Show Depth Overlay", variable=self._show_overlay_var)
        self._overlay_checkbutton.pack(anchor=tk.W)
        self._filter_checkbutton = ttk.Checkbutton(checkbox_frame, text="Apply Median Filter (Depth)", variable=self._apply_median_filter_var)
        self._filter_checkbutton.pack(anchor=tk.W)
        self._capture_btn = ttk.Button(controls_outer_frame, text="Capture Pose (Press 'W')", command=self._trigger_capture_async)
        self._capture_btn.pack(fill=tk.X, pady=(5,2))
        self.root.bind("<KeyPress-w>", lambda event: self._trigger_capture_async())
        self.root.bind("<KeyPress-W>", lambda event: self._trigger_capture_async())
        label_counts_frame = ttk.Labelframe(right_controls_and_counts_frame, text="Label Counts", padding="5")
        label_counts_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        for label_name_lc in POSE_LABELS:
            lbl_widget = ttk.Label(label_counts_frame, textvariable=self._label_counts_vars[label_name_lc])
            lbl_widget.pack(anchor=tk.W)
        status_bar_label = ttk.Label(main_frame, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar_label.pack(side=tk.BOTTOM, pady=(5,0), fill=tk.X)
        self._update_gui_status_message("GUI Initialized. Using CameraInfo for intrinsics.")


    def _unregister_ros_subscribers(self):
        if hasattr(self, 'timesync_filter') and self.timesync_filter is not None:
            self.timesync_filter.callbacks.clear() 
            if hasattr(self, 'rgb_subscriber_filter') and self.rgb_subscriber_filter is not None:
                self.rgb_subscriber_filter.sub.unregister()
                self.rgb_subscriber_filter = None
            if hasattr(self, 'depth_subscriber_filter') and self.depth_subscriber_filter is not None:
                self.depth_subscriber_filter.sub.unregister()
                self.depth_subscriber_filter = None
            if hasattr(self, 'info_subscriber_filter') and self.info_subscriber_filter is not None:
                self.info_subscriber_filter.sub.unregister()
                self.info_subscriber_filter = None
            self.timesync_filter = None 
        rospy.loginfo("ROS subscribers have been unregistered.")

    def _initialize_ros_topic_subscriptions(self):
        self._unregister_ros_subscribers() 
        try:
            rospy.loginfo(f"Initializing ROS subscribers. RGB Topic: {self.current_rgb_topic_to_subscribe}")
            self.rgb_subscriber_filter = message_filters.Subscriber(self.current_rgb_topic_to_subscribe, Image)
            self.depth_subscriber_filter = message_filters.Subscriber(REGISTERED_DEPTH_TOPIC, Image)
            self.info_subscriber_filter = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)

            self.timesync_filter = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_subscriber_filter, self.depth_subscriber_filter, self.info_subscriber_filter],
                queue_size=10, 
                slop=MESSAGE_FILTER_SLOP 
            )
            self.timesync_filter.registerCallback(self._ros_synchronized_callback)
            rospy.loginfo(f"Message filters synchronized for RGB: {self.current_rgb_topic_to_subscribe}, Depth, and CameraInfo.")
            self._gui_queue.put(GuiTask("update_status", "Waiting for synchronized ROS data..."))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)) 
        except Exception as e:
            rospy.logfatal(f"Failed to initialize ROS message_filters: {e}")
            self._gui_queue.put(GuiTask("update_status", "FATAL: Error setting up ROS Sync! Check topics."))
            self._gui_queue.put(GuiTask("set_controls_state", tk.DISABLED))

    def _on_apply_rgb_topic_change(self):
        selected_topic_display_name = self._rgb_topic_selection_var.get()
        new_selected_topic_path = RGB_TOPIC_OPTIONS.get(selected_topic_display_name)

        if new_selected_topic_path and new_selected_topic_path != self.current_rgb_topic_to_subscribe:
            rospy.loginfo(f"RGB topic change requested from '{self.current_rgb_topic_to_subscribe}' to '{new_selected_topic_path}'. Restarting subscribers.")
            self.current_rgb_topic_to_subscribe = new_selected_topic_path
            self._gui_queue.put(GuiTask("update_status", f"Changing RGB topic. Restarting subscribers..."))
            self._gui_queue.put(GuiTask("set_controls_state", tk.DISABLED))
            self.root.update_idletasks()
            self._initialize_ros_topic_subscriptions()
        elif not new_selected_topic_path:
            rospy.logwarn(f"Invalid RGB topic display name selected: {selected_topic_display_name}")
            messagebox.showwarning("Topic Error", "Invalid RGB topic selected.", parent=self.root)
        else:
            rospy.loginfo("Selected RGB topic is the same as current. No change needed.")

    def _ros_synchronized_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo): # SIMPLIFIED
        if self.is_shutting_down: return

        if not self._camera_info_topic_received : 
            try:
                if len(info_msg.K) == 9: 
                    fx, fy = info_msg.K[0], info_msg.K[4]
                    cx, cy = info_msg.K[2], info_msg.K[5]
                    if fx > 0 and fy > 0 and cx > 0 and cy > 0: 
                        self._fx_topic, self._fy_topic, self._cx_topic, self._cy_topic = fx, fy, cx, cy
                        self._camera_info_topic_received = True 
                        rospy.loginfo_once(f"Camera Info from TOPIC received: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                        self._gui_queue.put(GuiTask("update_status", "Ready for capture (using CameraInfo)."))
            except Exception as e:
                rospy.logerr_throttle(10, f"Error processing CameraInfo message from TOPIC: {e}")
        
        try:
            cv_rgb_image_raw = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth_image_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            image_for_pose_estimation = cv_rgb_image_raw.copy() 
            
            with self._data_access_lock:
                self._latest_synced_data["rgb"] = image_for_pose_estimation 
                self._latest_synced_data["depth"] = cv_depth_image_mm.copy() 
                self._latest_synced_data["stamp"] = rgb_msg.header.stamp
                self._latest_synced_data["info_msg"] = info_msg 

            overlay_display_image = self._create_overlay_image_for_gui(cv_rgb_image_raw, cv_depth_image_mm)
            self._gui_queue.put(GuiTask("update_main_preview", (cv_rgb_image_raw, overlay_display_image)))
            self._gui_queue.put(GuiTask("update_depth_preview", cv_depth_image_mm))

        except CvBridgeError as e:
            rospy.logerr_throttle(5, f"CvBridge error in synchronized_callback: {e}")
        except Exception as e:
            rospy.logerr_throttle(5, f"Error processing synchronized messages: {e}")

    def _process_capture_and_get_label(self): 
        current_metadata_for_capture = PendingCaptureMetadata()
        current_metadata_for_capture.timestamp = datetime.datetime.now().isoformat()
        current_metadata_for_capture.rgb_source_topic = self.current_rgb_topic_to_subscribe
        current_metadata_for_capture.calibration_used = False
        current_metadata_for_capture.median_filter_applied = self._apply_median_filter_var.get()
        current_metadata_for_capture.openpose_conf_threshold_value = OPENPOSE_CONFIDENCE_THRESHOLD
        
        rospy.loginfo(f"Capture triggered. Median Filter: {current_metadata_for_capture.median_filter_applied}")

        with self._data_access_lock:
            rgb_image_for_pose = self._latest_synced_data["rgb"] 
            depth_image_mm_raw = self._latest_synced_data["depth"]
            # camera_info_msg_current = self._latest_synced_data["info_msg"]

        if rgb_image_for_pose is None or depth_image_mm_raw is None:
            rospy.logwarn("Capture aborted: Missing RGB or Depth frame from synced data.")
            self._gui_queue.put(GuiTask("update_status", "Error: Missing image data for capture."))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        if not self._camera_info_topic_received or self._fx_topic is None:
            rospy.logerr("Capture critical error: Camera intrinsics from /camera_info not yet received.")
            self._gui_queue.put(GuiTask("update_status", "CRITICAL: No intrinsics from CameraInfo!"))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return
        
        fx_proj, fy_proj, cx_proj, cy_proj = self._fx_topic, self._fy_topic, self._cx_topic, self._cy_topic
        intrinsics_source_info_str = "CAM_INFO_TOPIC"
        
        self._gui_queue.put(GuiTask("update_status", f"Detecting hand (Intrinsics: {intrinsics_source_info_str})..."))

        depth_image_for_processing = depth_image_mm_raw.copy()
        filter_applied_log_msg = "RawDepth"
        if current_metadata_for_capture.median_filter_applied:
            try:
                k_size = MEDIAN_FILTER_KERNEL_SIZE; k_size = k_size + 1 if k_size % 2 == 0 else k_size
                depth_image_for_processing = cv2.medianBlur(depth_image_mm_raw, k_size)
                filter_applied_log_msg = f"MedianDepth(k={k_size})"
            except Exception as e_filter:
                rospy.logerr(f"Error applying Median filter: {e_filter}. Using raw depth.")
        
        try:
            all_peaks_2d_from_openpose = self._hand_estimator(rgb_image_for_pose)
        except Exception as e_pose:
            rospy.logerr(f"Error during OpenPose hand estimation: {e_pose}")
            self._gui_queue.put(GuiTask("update_status", "Hand Estimation Error!"))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        if not isinstance(all_peaks_2d_from_openpose, np.ndarray) or all_peaks_2d_from_openpose.ndim != 2:
            rospy.logwarn(f"OpenPose output not a 2D NumPy array. Type: {type(all_peaks_2d_from_openpose)}")
            self._gui_queue.put(GuiTask("update_status", "Hand not detected or invalid OpenPose output."))
            self._gui_queue.put(GuiTask("clear_2d_pose", "No Pose Detected"))
            self._gui_queue.put(GuiTask("clear_3d_plot", "No Pose Detected"))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        current_metadata_for_capture.num_2d_peaks_detected_raw = all_peaks_2d_from_openpose.shape[0]
        
        peaks_2d_coords_after_conf_filter_list = []
        original_indices_of_confident_peaks = []
        self._pending_capture_data_instance.peaks_2d_filtered = np.full((21,2), np.nan, dtype=np.float32)
        
        has_confidence = all_peaks_2d_from_openpose.shape[1] >= 3
        if not has_confidence:
            current_metadata_for_capture.openpose_conf_threshold_value = -1.0
            for i_idx in range(min(21, current_metadata_for_capture.num_2d_peaks_detected_raw)):
                peaks_2d_coords_after_conf_filter_list.append(all_peaks_2d_from_openpose[i_idx, :2])
                original_indices_of_confident_peaks.append(i_idx)
                if i_idx < 21: self._pending_capture_data_instance.peaks_2d_filtered[i_idx] = all_peaks_2d_from_openpose[i_idx, :2]
        else:
            for i_idx in range(min(21, current_metadata_for_capture.num_2d_peaks_detected_raw)):
                if all_peaks_2d_from_openpose[i_idx, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD:
                    coords_2d = all_peaks_2d_from_openpose[i_idx, :2]
                    peaks_2d_coords_after_conf_filter_list.append(coords_2d)
                    original_indices_of_confident_peaks.append(i_idx)
                    if i_idx < 21: self._pending_capture_data_instance.peaks_2d_filtered[i_idx] = coords_2d
        
        current_metadata_for_capture.num_2d_peaks_above_conf = len(peaks_2d_coords_after_conf_filter_list)

        if current_metadata_for_capture.num_2d_peaks_above_conf == 0:
            status_msg = f"No 2D keypoints above conf ({OPENPOSE_CONFIDENCE_THRESHOLD}). Raw: {current_metadata_for_capture.num_2d_peaks_detected_raw}. Try Again."
            self._gui_queue.put(GuiTask("update_pose_preview", (rgb_image_for_pose.copy(), [False]*21 )))
            self._gui_queue.put(GuiTask("clear_3d_plot", "No Reliable 2D Keypoints"))
            self._gui_queue.put(GuiTask("update_status", status_msg))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return
        
        keypoints_in_camera_frame_3d_raw = np.full((current_metadata_for_capture.num_2d_peaks_above_conf, 3), np.nan, dtype=np.float32)
        validity_mask_after_3d_projection = [False] * current_metadata_for_capture.num_2d_peaks_above_conf

        for i_conf_peak, coords_2d_conf in enumerate(peaks_2d_coords_after_conf_filter_list):
            x_pixel, y_pixel = float(coords_2d_conf[0]), float(coords_2d_conf[1])
            depth_meters = self._get_depth_from_neighborhood_pixels(depth_image_for_processing, x_pixel, y_pixel, DEPTH_NEIGHBORHOOD_SIZE)
            if not np.isnan(depth_meters):
                x_cam_coord = (x_pixel - cx_proj) * depth_meters / fx_proj
                y_cam_coord = (y_pixel - cy_proj) * depth_meters / fy_proj
                keypoints_in_camera_frame_3d_raw[i_conf_peak] = [x_cam_coord, y_cam_coord, depth_meters]
                validity_mask_after_3d_projection[i_conf_peak] = True
        
        current_metadata_for_capture.num_3d_points_initial = np.sum(validity_mask_after_3d_projection)

        keypoints_in_camera_frame_3d_full_skeleton = np.full((21, 3), np.nan, dtype=np.float32)
        for i_reconstruct, original_skeleton_idx in enumerate(original_indices_of_confident_peaks):
            if original_skeleton_idx < 21 and validity_mask_after_3d_projection[i_reconstruct]:
                keypoints_in_camera_frame_3d_full_skeleton[original_skeleton_idx] = keypoints_in_camera_frame_3d_raw[i_reconstruct]
        
        keypoints_relative_to_wrist_3d = np.full_like(keypoints_in_camera_frame_3d_full_skeleton, np.nan)
        if not np.isnan(keypoints_in_camera_frame_3d_full_skeleton[0]).any():
            wrist_3d_in_camera_frame = keypoints_in_camera_frame_3d_full_skeleton[0].copy()
            keypoints_relative_to_wrist_3d = keypoints_in_camera_frame_3d_full_skeleton - wrist_3d_in_camera_frame
            keypoints_relative_to_wrist_3d[0] = [0.0, 0.0, 0.0]
        else:
            rospy.logwarn("Wrist (keypoint 0) has no valid 3D projection. Relative coordinates will be based on NaNs.")
            keypoints_relative_to_wrist_3d = keypoints_in_camera_frame_3d_full_skeleton

        keypoints_relative_3d_final_filtered, final_validity_mask_for_saving = self._filter_3d_keypoints_by_distance(keypoints_relative_to_wrist_3d)
        num_3d_points_final_count = np.sum(final_validity_mask_for_saving)

        self._pending_capture_data_instance.capture_metadata = current_metadata_for_capture
        self._pending_capture_data_instance.validity_mask_final = final_validity_mask_for_saving

        self._gui_queue.put(GuiTask("update_pose_preview", (rgb_image_for_pose.copy(), final_validity_mask_for_saving)))
        plot_title_3d = f"Relative 3D ({num_3d_points_final_count} valid, Intr: {intrinsics_source_info_str}, Depth: {filter_applied_log_msg})"
        self._gui_queue.put(GuiTask("update_3d_plot", (keypoints_relative_3d_final_filtered, plot_title_3d)))

        if num_3d_points_final_count < MIN_VALID_KEYPOINTS_FOR_SAVE:
            status_msg = f"Only {num_3d_points_final_count} valid 3D points (min {MIN_VALID_KEYPOINTS_FOR_SAVE}). Press W."
            self._gui_queue.put(GuiTask("update_status", status_msg))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return
        
        if not final_validity_mask_for_saving[0]:
            self._gui_queue.put(GuiTask("update_status", "Wrist invalid after filters! Press W."))
            self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL)); return

        self._pending_capture_data_instance.keypoints_rel_3d = keypoints_relative_3d_final_filtered.copy()
        self._waiting_for_label_input = True

        label_keys_prompt_parts = [f"{str(i+1) if i < 9 else ('O' if POSE_LABELS[i].lower() == 'other' else str(i+1))}={POSE_LABELS[i]}" for i in range(len(POSE_LABELS))]
        status_prompt_for_label = f"{num_3d_points_final_count} valid 3D pts. Label: [{', '.join(label_keys_prompt_parts)}] or Q=Cancel."
        self._gui_queue.put(GuiTask("update_status", status_prompt_for_label))

    def _handle_label_key_press(self, label_idx: int):
        if not self._waiting_for_label_input: return
        if 0 <= label_idx < len(POSE_LABELS):
            selected_label = POSE_LABELS[label_idx]
            rospy.loginfo(f"Label key '{selected_label}' (index {label_idx}) pressed. Saving current pose...")
            self._save_pending_capture_to_csv(selected_label)
            self._label_counts[selected_label] = self._label_counts.get(selected_label, 0) + 1
            self._gui_queue.put(GuiTask("update_label_counts_display", None))
        else:
            rospy.logwarn(f"Invalid label index received from key press: {label_idx}")
        self._waiting_for_label_input = False
        self._pending_capture_data_instance = CaptureData()
        self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL))
        self._gui_queue.put(GuiTask("update_status", "Ready for next capture (Press W)."))

    def _handle_cancel_key_press(self, event=None):
        if not self._waiting_for_label_input: return
        rospy.loginfo("Cancel key 'Q' pressed. Discarding current capture.")
        self._cancel_pending_capture_and_save()

    def _cancel_pending_capture_and_save(self):
        self._waiting_for_label_input = False
        self._pending_capture_data_instance = CaptureData()
        self._gui_queue.put(GuiTask("update_status", "Capture cancelled by user. Ready."))
        self._gui_queue.put(GuiTask("clear_3d_plot", "Capture Cancelled"))
        self._gui_queue.put(GuiTask("clear_2d_pose", "Capture Cancelled"))
        self._gui_queue.put(GuiTask("set_controls_state", tk.NORMAL))

    def _save_pending_capture_to_csv(self, assigned_label: str):
        data_to_write = self._pending_capture_data_instance
        points_to_write_rel_3d = data_to_write.keypoints_rel_3d
        metadata_to_write = data_to_write.capture_metadata

        if points_to_write_rel_3d is None or metadata_to_write is None:
            rospy.logerr("Attempted to save, but no keypoints or metadata available!")
            self._gui_queue.put(GuiTask("update_status", "Error: No data to save!"))
            return

        num_valid_3d_points_to_save = np.sum(data_to_write.validity_mask_final) if data_to_write.validity_mask_final is not None else 0
        self._gui_queue.put(GuiTask("update_status", f"Saving {num_valid_3d_points_to_save} valid points as '{assigned_label}'..."))

        try:
            with open(CSV_FULL_PATH, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                row_to_write = [
                    assigned_label, 
                    metadata_to_write.timestamp, 
                    metadata_to_write.rgb_source_topic, 
                    False,
                    metadata_to_write.median_filter_applied, 
                    metadata_to_write.openpose_conf_threshold_value,
                    metadata_to_write.num_2d_peaks_detected_raw, 
                    metadata_to_write.num_2d_peaks_above_conf,
                    metadata_to_write.num_3d_points_initial, 
                    num_valid_3d_points_to_save
                ]
                flattened_coords_for_csv = []
                for i_coord_set in range(21):
                    if data_to_write.validity_mask_final and \
                       i_coord_set < len(data_to_write.validity_mask_final) and \
                       data_to_write.validity_mask_final[i_coord_set] and \
                       i_coord_set < points_to_write_rel_3d.shape[0] and \
                       not np.isnan(points_to_write_rel_3d[i_coord_set]).any():
                        flattened_coords_for_csv.extend([f"{coord:.6f}" for coord in points_to_write_rel_3d[i_coord_set]])
                    else:
                        flattened_coords_for_csv.extend(["", "", ""]) 
                row_to_write.extend(flattened_coords_for_csv)
                writer.writerow(row_to_write)
            self._gui_queue.put(GuiTask("update_status", f"Saved pose '{assigned_label}' ({num_valid_3d_points_to_save} valid pts). Ready."))
            rospy.loginfo(f"Pose '{assigned_label}' data appended to {CSV_FULL_PATH}")
        except IOError as e_io:
            rospy.logerr(f"IOError writing to CSV '{CSV_FULL_PATH}': {e_io}")
            self._gui_queue.put(GuiTask("update_status", f"Error writing CSV: {e_io}"))
        except Exception as e_gen_save:
            rospy.logerr(f"Unexpected error saving data to CSV: {e_gen_save}")
            self._gui_queue.put(GuiTask("update_status", "Error saving CSV! Check logs."))

    def run_gui_mainloop(self):
        try:
            rospy.loginfo("Starting Tkinter GUI main loop...")
            self.root.mainloop()
        except Exception as e_gui_loop:
            rospy.logerr(f"Error in Tkinter GUI main loop: {e_gui_loop}")
        finally:
            rospy.loginfo("Exiting Tkinter GUI main loop.")
            self.on_close()

    def ros_spin_thread_target(self):
        try:
            rospy.spin()
            rospy.loginfo("ROS spin thread finished.")
        except rospy.ROSInterruptException:
            rospy.loginfo("ROS spin thread interrupted (ROS shutdown).")
        except Exception as e_ros_spin:
            rospy.logerr(f"Error in ROS spin thread: {e_ros_spin}")

    def on_close(self):
        if not self.is_shutting_down:
            rospy.loginfo("GUI window close requested. Initiating shutdown sequence...")
            self.is_shutting_down = True
            if self._after_id_gui_pump:
                try: self.root.after_cancel(self._after_id_gui_pump)
                except tk.TclError: pass
                self._after_id_gui_pump = None
            self._unregister_ros_subscribers()
            if not rospy.is_shutdown():
                rospy.loginfo("Signaling ROS shutdown...")
                rospy.signal_shutdown("GUI closed by user")
            try: 
                if hasattr(self, '_fig_3d') and plt.fignum_exists(self._fig_3d.number): 
                    plt.close(self._fig_3d) 
            except Exception as e_plt: rospy.logwarn(f"Error closing matplotlib figure: {e_plt}")
            try: 
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.quit(); self.root.destroy()
            except tk.TclError: pass 
            except Exception as e_tk: rospy.logwarn(f"Error destroying Tkinter root window: {e_tk}")
            rospy.loginfo("GUI resources released. Collector shut down.")

    def _process_gui_queue(self):
        if self.is_shutting_down: return
        current_task: Optional[GuiTask] = None
        try:
            while not self._gui_queue.empty():
                current_task = self._gui_queue.get_nowait()
                command, payload = current_task.command, current_task.payload
                if not hasattr(self, 'root') or not self.root.winfo_exists(): break
                if command == "update_status":
                    if hasattr(self, '_status_var'): self._status_var.set(str(payload))
                elif command == "update_main_preview":
                    if payload is not None: self._update_main_preview_display(payload)
                elif command == "update_depth_preview":
                    if payload is not None: self._update_depth_preview_display(payload)
                elif command == "clear_depth_preview":
                    self._clear_depth_preview_display(str(payload) if payload is not None else "")
                elif command == "update_pose_preview":
                    if payload is not None: self._update_2d_pose_preview_display(payload)
                elif command == "clear_2d_pose":
                    self._clear_2d_pose_preview_display(str(payload) if payload is not None else "")
                elif command == "update_3d_plot":
                    if payload is not None and len(payload) == 2: self._update_3d_plot_display(payload[0], str(payload[1]))
                elif command == "clear_3d_plot":
                    self._clear_3d_plot_display(str(payload) if payload is not None else "")
                elif command == "set_controls_state":
                    self._set_gui_controls_state_internal(payload if payload in [tk.NORMAL, tk.DISABLED] else tk.NORMAL)
                elif command == "update_label_counts_display":
                    self._update_label_counts_gui_display()
                else: rospy.logwarn(f"Unknown GUI command received: {command}")
                self._gui_queue.task_done()
        except queue.Empty: pass
        except Exception as e_pump:
            command_name_log = current_task.command if current_task else "UnknownTask"
            rospy.logerr(f"Error processing GUI queue (Command: {command_name_log}): {e_pump}")
        finally:
            if not self.is_shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
                self._after_id_gui_pump = self.root.after(50, self._process_gui_queue)

    def _set_gui_controls_state_internal(self, target_tk_state: str):
        if hasattr(self, '_capture_btn'): self._capture_btn.config(state=target_tk_state)
        if hasattr(self, '_filter_checkbutton'): self._filter_checkbutton.config(state=target_tk_state)
        if hasattr(self, '_overlay_checkbutton'): self._overlay_checkbutton.config(state=target_tk_state)
        if hasattr(self, '_rgb_topic_dropdown'): self._rgb_topic_dropdown.config(state="readonly")
        if hasattr(self, '_apply_topic_btn'): self._apply_topic_btn.config(state=tk.NORMAL if target_tk_state == tk.NORMAL else tk.DISABLED)

    def _update_label_counts_gui_display(self):
        for label_key, count_str_var in self._label_counts_vars.items():
            count_str_var.set(f"{label_key}: {self._label_counts.get(label_key, 0)}")

    def _resize_image_for_display(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_h, max_w = DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH
        if h > max_h or w > max_w:
            scale_factor = min(max_h / h, max_w / w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image

    def _create_overlay_image_for_gui(self, rgb_image_bgr: np.ndarray, depth_image_16uc1: np.ndarray) -> Optional[np.ndarray]: # (Identical)
        if rgb_image_bgr is None or depth_image_16uc1 is None: return None
        try:
            min_d_vis, max_d_vis_thresh = VALID_DEPTH_THRESHOLD_MM
            depth_for_vis = depth_image_16uc1.copy()
            depth_for_vis[depth_for_vis < min_d_vis] = min_d_vis 
            depth_for_vis[depth_for_vis > max_d_vis_thresh] = max_d_vis_thresh
            normalized_depth_vis = cv2.normalize(depth_for_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colorized_depth_vis = cv2.applyColorMap(normalized_depth_vis, cv2.COLORMAP_JET)
            h_rgb, w_rgb = rgb_image_bgr.shape[:2]; h_depth, w_depth = colorized_depth_vis.shape[:2]
            if h_rgb != h_depth or w_rgb != w_depth:
                colorized_depth_resized_vis = cv2.resize(colorized_depth_vis, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            else: colorized_depth_resized_vis = colorized_depth_vis
            return cv2.addWeighted(rgb_image_bgr, OVERLAY_ALPHA, colorized_depth_resized_vis, 1.0 - OVERLAY_ALPHA, 0.0)
        except Exception as e:
            rospy.logerr_throttle(5, f"Error creating overlay image for GUI: {e}")
            return rgb_image_bgr

    def _get_depth_from_neighborhood_pixels(self, depth_map_mm: np.ndarray, center_x_px: float, center_y_px: float, neighborhood_kernel_size: int = 3) -> float: # (Identical)
        if depth_map_mm is None: return np.nan
        if neighborhood_kernel_size % 2 == 0: neighborhood_kernel_size += 1
        radius_px = neighborhood_kernel_size // 2
        map_h, map_w = depth_map_mm.shape
        ix_center, iy_center = int(round(center_x_px)), int(round(center_y_px))
        if not (0 <= ix_center < map_w and 0 <= iy_center < map_h): return np.nan
        y_start, y_end = max(0, iy_center - radius_px), min(map_h, iy_center + radius_px + 1)
        x_start, x_end = max(0, ix_center - radius_px), min(map_w, ix_center + radius_px + 1)
        neighborhood_pixels = depth_map_mm[y_start:y_end, x_start:x_end]
        min_depth_mm, max_depth_mm_thresh = VALID_DEPTH_THRESHOLD_MM
        valid_depths_in_neighborhood_mm = neighborhood_pixels[(neighborhood_pixels >= min_depth_mm) & (neighborhood_pixels <= max_depth_mm_thresh)]
        if valid_depths_in_neighborhood_mm.size < max(1, (neighborhood_kernel_size**2) // 4): return np.nan
        std_dev_of_valid_depths_mm = np.std(valid_depths_in_neighborhood_mm)
        if std_dev_of_valid_depths_mm > DEPTH_STD_DEV_THRESHOLD_MM:
            rospy.logdebug_throttle(1,f"High std dev ({std_dev_of_valid_depths_mm:.1f}mm) for ({center_x_px:.0f},{center_y_px:.0f}) in depth neighborhood. Returning NaN.")
            return np.nan
        median_depth_mm = np.median(valid_depths_in_neighborhood_mm)
        return float(median_depth_mm / 1000.0)

    def _filter_3d_keypoints_by_distance(self, keypoints_3d_relative_coords: np.ndarray) -> Tuple[np.ndarray, List[bool]]: # (Identical)
        if keypoints_3d_relative_coords is None or keypoints_3d_relative_coords.shape[0] != 21:
            rospy.logwarn(f"[_filter_3d_outliers] Input keypoints_3d_relative_coords is invalid. Got shape: {keypoints_3d_relative_coords.shape if keypoints_3d_relative_coords is not None else 'None'}")
            return np.full((21, 3), np.nan, dtype=np.float32), [False] * 21
        points_after_filter = keypoints_3d_relative_coords.copy()
        current_validity_mask = ~np.isnan(points_after_filter).any(axis=1)
        if not current_validity_mask[0]:
             rospy.logwarn_throttle(5,"Wrist keypoint (0) is invalid/NaN before distance filtering. Cannot filter relative to it.")
             return points_after_filter, current_validity_mask.tolist()
        max_distance_squared = OUTLIER_XYZ_THRESHOLD_M ** 2
        num_points_removed_by_filter = 0
        for i in range(1, 21):
            if current_validity_mask[i]:
                distance_sq_from_wrist = np.sum(points_after_filter[i]**2) 
                if distance_sq_from_wrist > max_distance_squared:
                    points_after_filter[i] = np.nan
                    current_validity_mask[i] = False
                    num_points_removed_by_filter += 1
        if num_points_removed_by_filter > 0:
            rospy.loginfo(f"Removed {num_points_removed_by_filter} 3D outlier points (distance from wrist > {OUTLIER_XYZ_THRESHOLD_M:.2f}m).")
        return points_after_filter, current_validity_mask.tolist()

if __name__ == "__main__":
    data_collector_instance = None
    try:
        data_collector_instance = DepthHandCollector()
        data_collector_instance.run_gui_mainloop()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted by ROS master (e.g., Ctrl+C in terminal).")
    except KeyboardInterrupt: 
        rospy.loginfo("Script interrupted by user (Ctrl+C).")
    except SystemExit as e_sys_exit: 
        rospy.logfatal(f"System exit called, likely due to critical initialization error: {e_sys_exit}")
    except Exception as e_main:
        rospy.logfatal(f"Unhandled exception in main execution block: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if data_collector_instance and hasattr(data_collector_instance, 'on_close') and not data_collector_instance.is_shutting_down:
            rospy.loginfo("Performing final cleanup in main's finally block...")
            data_collector_instance.on_close()
        if not rospy.is_shutdown():
            rospy.loginfo("ROS is not shutdown in main finally block, signaling shutdown now.")
            rospy.signal_shutdown("Script main block finished or unhandled error.")
        rospy.loginfo("DepthHandCollector script has finished execution.")
        print("\nExiting DepthHandCollector script.")