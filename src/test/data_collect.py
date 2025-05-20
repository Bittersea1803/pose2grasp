import os
import sys
import time
import queue
import threading
import csv
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import yaml
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

OPENPOSE_REPO = "/home/openpose_user/src/pose2grasp/src/pytorch-openpose"
try:
    if not os.path.isdir(OPENPOSE_REPO):
        raise ImportError(f"OpenPose repo directory not found at: {OPENPOSE_REPO}")
    sys.path.append(OPENPOSE_REPO)
    if not os.path.exists(os.path.join(OPENPOSE_REPO, "src", "__init__.py")):
        from hand import Hand
        import util as openpose_util 
        rospy.logwarn("Importing OpenPose directly, assuming src is not a package.")
    else:
        from src.hand import Hand
        from src import util as openpose_util
except ImportError as e:
    print(f"Error: Cannot import OpenPose from {OPENPOSE_REPO}.")
    print(f"Check the path and ensure the library is correctly installed/built.")
    print(f"ImportError: {e}")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)


RGB_SOURCE_SELECTION_DEFAULT = "rect_color"
RGB_TOPIC_RAW = "/camera/rgb/image_color"
RGB_TOPIC_RECT_COLOR = "/camera/rgb/image_rect_color"
RGB_TOPIC_OPTIONS = {
    "Rectified Color (`image_rect_color`)": RGB_TOPIC_RECT_COLOR,
    "Raw Color (`image_color`)": RGB_TOPIC_RAW
}


REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

SAVE_ROOT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CALIBRATION_FILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CALIBRATION_FILE_NAME = "calibration_camera_rgb_image_rect_color.yaml"
CALIBRATION_FILE_PATH = os.path.join(CALIBRATION_FILE_DIR, CALIBRATION_FILE_NAME)

CSV_FILENAME = "collected_hand_poses_metadata.csv"
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

os.makedirs(SAVE_ROOT_CSV, exist_ok=True)

hand_connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]
limb_colors = [
    [255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],
    [0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],[0,0,255],[85,0,255],
    [170,0,255],[255,0,255],[255,0,170],[255,0,85],[85,85,85],[170,170,170]
]
VALID_POINT_COLOR = (0, 255, 0); INVALID_POINT_COLOR = (0, 0, 255)

@dataclass
class PendingCaptureMetadata:
    timestamp: Optional[str] = None
    rgb_source_topic: str = ""
    calibration_used: bool = False
    median_filter_applied: bool = False
    openpose_conf_threshold_value: float = 0.0
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
        rospy.init_node("hand_collector_enhanced", anonymous=True)
        self.bridge = CvBridge()
        rospy.loginfo("ROS Node Initialized.")
        self.is_shutting_down = False
        self._after_id: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("Hand Collector (Enhanced + Metadata)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._gui_queue: queue.Queue[GuiTask] = queue.Queue()

        self._status_var = tk.StringVar(value="Initializing...")
        self._apply_median_filter_var = tk.BooleanVar(value=True)
        self._show_overlay_var = tk.BooleanVar(value=True)
        self._use_calibration_var = tk.BooleanVar(value=False)
        self._label_counts_vars: Dict[str, tk.StringVar] = {}
        self._rgb_topic_selection_var = tk.StringVar()

        self._K_calib: Optional[np.ndarray] = None
        self._D_calib: Optional[np.ndarray] = None
        self._calib_img_width: Optional[int] = None
        self._calib_img_height: Optional[int] = None
        self._calibration_loaded_successfully = False
        self._calibration_active = False

        self.current_rgb_topic_to_subscribe = RGB_TOPIC_RECT_COLOR
        if RGB_SOURCE_SELECTION_DEFAULT.lower() == "raw":
            self.current_rgb_topic_to_subscribe = RGB_TOPIC_RAW

        default_dropdown_key = ""
        for key, value in RGB_TOPIC_OPTIONS.items():
            if value == self.current_rgb_topic_to_subscribe:
                default_dropdown_key = key
                break
        self._rgb_topic_selection_var.set(default_dropdown_key or list(RGB_TOPIC_OPTIONS.keys())[0])


        self._initialize_openpose()
        self._initialize_data_storage()
        self._initialize_camera_params_from_topic()
        self._load_calibration_file()
        self._initialize_csv()
        self._initialize_label_counts()
        self._setup_gui_layout()
        self._initialize_ros_subscriptions()

        self._waiting_for_label = False

        for i, label in enumerate(POSE_LABELS):
            key = str(i + 1)
            self.root.bind(f"<KeyPress-{key}>", lambda event, idx=i: self._handle_label_key(idx))
        if "other" in POSE_LABELS:
            other_idx = POSE_LABELS.index("other")
            self.root.bind("<KeyPress-o>", lambda event, idx=other_idx: self._handle_label_key(idx))
            self.root.bind("<KeyPress-O>", lambda event, idx=other_idx: self._handle_label_key(idx))

        self.root.bind("<KeyPress-q>", self._handle_cancel_key)
        self.root.bind("<KeyPress-Q>", self._handle_cancel_key)

        self._pump_gui_queue()

        if not rospy.is_shutdown():
            threading.Thread(target=self.ros_spin_target, daemon=True).start()

    def _initialize_openpose(self):
        try:
            model_path = os.path.join(OPENPOSE_REPO, "model", "hand_pose_model.pth")
            if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found: {model_path}")
            self._hand_estimator = Hand(model_path)
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            rospy.loginfo(f"[Collector] OpenPose using device: {self._device}")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize OpenPose: {e}")
            parent_window = self.root if hasattr(self, 'root') and self.root.winfo_exists() else None
            messagebox.showerror("Fatal Error", f"Failed to initialize OpenPose: {e}", parent=parent_window)
            sys.exit(1)

    def _initialize_label_counts(self):
        self._label_counts: Dict[str, int] = {label: 0 for label in POSE_LABELS}

        if hasattr(self, '_csv_path') and self._csv_path and \
            os.path.exists(self._csv_path) and os.path.getsize(self._csv_path) > 0:
            try:
                with open(self._csv_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)

                    if 'label' not in reader.fieldnames:
                        rospy.logwarn(
                            f"CSV datoteka '{self._csv_path}' ne sadrži stupac 'label' u zaglavlju. "
                            "Nije moguće učitati postojeće brojače."
                        )
                    else:
                        rospy.loginfo(f"Učitavanje postojećih brojača oznaka iz '{self._csv_path}'...")
                        loaded_counts_temp = {label: 0 for label in POSE_LABELS}
                        for row in reader:
                            label_from_csv = row.get('label')
                            if label_from_csv in loaded_counts_temp:
                                loaded_counts_temp[label_from_csv] += 1
                            elif label_from_csv:
                                rospy.logwarn_once(
                                    f"Oznaka '{label_from_csv}' iz CSV datoteke nije definirana u POSE_LABELS. "
                                    "Bit će ignorirana za brojače u trenutnoj sesiji."
                                )
                        for lbl_key in self._label_counts:
                            if lbl_key in loaded_counts_temp:
                                self._label_counts[lbl_key] = loaded_counts_temp[lbl_key]

                        rospy.loginfo(f"Završeno učitavanje brojača iz CSV-a: {self._label_counts}")
            except Exception as e:
                rospy.logerr(f"Greška pri čitanju CSV datoteke ('{self._csv_path}') za popunjavanje brojača oznaka: {e}. "
                                "Brojači će krenuti od 0 za ovu sesiju.")
                self._label_counts = {label: 0 for label in POSE_LABELS}
        else:
            rospy.loginfo("Postojeća CSV datoteka nije pronađena, prazna je ili _csv_path nije postavljen. "
                            "Brojači oznaka će krenuti od 0.")

        for label_text in POSE_LABELS:
            current_count = self._label_counts.get(label_text, 0)
            self._label_counts_vars[label_text] = tk.StringVar(value=f"{label_text}: {current_count}")

        if hasattr(self, 'root') and self.root.winfo_exists() and hasattr(self, '_update_label_counts_display'):
            self._gui_queue.put(GuiTask("update_label_counts", None))

    def _initialize_data_storage(self):
        self._latest_synced_data: Dict[str, Any] = {"rgb": None, "depth": None, "stamp": None}
        self._data_lock = threading.Lock()
        self._pending_capture_data = CaptureData()

    def _initialize_camera_params_from_topic(self):
        self._fx_topic: Optional[float] = None; self._fy_topic: Optional[float] = None
        self._cx_topic: Optional[float] = None; self._cy_topic: Optional[float] = None
        self._camera_info_topic_received = False

    def _load_calibration_file(self):
        rospy.loginfo(f"Attempting to load calibration file from: {CALIBRATION_FILE_PATH}")
        self._calibration_loaded_successfully = False
        self._K_calib = None
        self._D_calib = None
        self._calib_img_width = None
        self._calib_img_height = None

        try:
            if os.path.exists(CALIBRATION_FILE_PATH):
                with open(CALIBRATION_FILE_PATH, 'r') as f:
                    calib_data = yaml.safe_load(f)

                if 'camera_matrix' not in calib_data:
                    raise KeyError("Ključ 'camera_matrix' nije pronađen u YAML datoteci.")
                cm_data_raw = calib_data['camera_matrix']
                if isinstance(cm_data_raw, dict) and 'data' in cm_data_raw and \
                    'rows' in cm_data_raw and 'cols' in cm_data_raw:
                    self._K_calib = np.array(cm_data_raw['data']).reshape(cm_data_raw['rows'], cm_data_raw['cols'])
                elif isinstance(cm_data_raw, list):
                    self._K_calib = np.array(cm_data_raw)
                    if self._K_calib.shape != (3,3):
                        raise ValueError(f"Camera matrix iz liste ima neispravan oblik: {self._K_calib.shape}. Očekivano (3,3).")
                else:
                    raise ValueError("Format camera_matrix nije prepoznat.")

                dc_key_to_try = 'distortion_coefficients'
                if dc_key_to_try not in calib_data:
                    dc_key_to_try = 'dist_coeffs'
                    if dc_key_to_try not in calib_data:
                        raise KeyError("Distorzijski koeficijenti nisu pronađeni (provjereni ključevi 'distortion_coefficients' i 'dist_coeffs').")

                dc_data_raw = calib_data[dc_key_to_try]
                if isinstance(dc_data_raw, dict) and 'data' in dc_data_raw and \
                    'rows' in dc_data_raw and 'cols' in dc_data_raw:
                    self._D_calib = np.array(dc_data_raw['data']).reshape(dc_data_raw['rows'], dc_data_raw['cols'])
                elif isinstance(dc_data_raw, list):
                    self._D_calib = np.array(dc_data_raw)
                    if self._D_calib.ndim == 2 and self._D_calib.shape[0] == 1:
                        pass
                    elif self._D_calib.ndim == 1:
                        self._D_calib = self._D_calib.reshape(1, -1)
                    else:
                        try:
                            temp_coeffs = [item[0] for item in dc_data_raw if isinstance(item, list) and len(item)==1]
                            if len(temp_coeffs) == len(dc_data_raw) and len(temp_coeffs) >= 4:
                                self._D_calib = np.array(temp_coeffs).reshape(1, -1)
                            else: raise ValueError()
                        except:
                            raise ValueError(f"Distorzijski koeficijenti iz liste imaju nepodržanu strukturu: {self._D_calib.shape if self._D_calib is not None else 'nepoznat oblik'}")

                    if self._D_calib.shape[0] !=1 or self._D_calib.shape[1] < 4:
                        raise ValueError(f"Distorzijski koeficijenti trebaju biti oblika (1,N) s barem 4 vrijednosti, dobiveno {self._D_calib.shape}.")
                else:
                    raise ValueError("Format distorzijskih koeficijenata nije prepoznat.")

                if 'image_width' not in calib_data or 'image_height' not in calib_data:
                    rospy.logwarn("image_width ili image_height nisu pronađeni u YAML datoteci. "
                                    "Preskačem provjeru rezolucije pri undistortionu.")
                    self._calib_img_width = None
                    self._calib_img_height = None
                else:
                    self._calib_img_width = int(calib_data['image_width'])
                    self._calib_img_height = int(calib_data['image_height'])

                rospy.loginfo(f"Kalibracijska datoteka '{CALIBRATION_FILE_PATH}' uspješno učitana i parsirana.")
                self._calibration_loaded_successfully = True
                self._update_gui_status("Kalibracijska datoteka učitana. Odaberi kućicu za korištenje.")
            else:
                rospy.logwarn(f"Kalibracijska datoteka nije pronađena na putanji: {CALIBRATION_FILE_PATH}.")
                self._update_gui_status("Kalibracijska datoteka nije pronađena. Koristi se CameraInfo topic.")
        except Exception as e:
            rospy.logerr(f"Greška pri učitavanju ili parsiranju kalibracijske datoteke '{CALIBRATION_FILE_PATH}': {e}")
            self._update_gui_status(f"Greška pri učitavanju kalibracije: {e}")
            self._K_calib = None
            self._D_calib = None
            import traceback
            rospy.logerr(f"Detalji greške pri učitavanju kalibracije:\n{traceback.format_exc()}")

        if hasattr(self, '_calib_checkbox'):
            if self.root.winfo_exists():
                self._calib_checkbox.config(state=tk.NORMAL if self._calibration_loaded_successfully else tk.DISABLED)

        if hasattr(self, 'root') and self.root.winfo_exists():
            self._on_toggle_calibration()

    def _setup_gui_layout(self):
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
        self._clear_depth_preview("Waiting for Depth...")

        right_vis_col = ttk.Frame(vis_frame)
        right_vis_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._pose_2d_label = ttk.Label(right_vis_col, text="Waiting for 2D Pose...")
        self._pose_2d_label.pack(pady=1, fill=tk.BOTH, expand=True)
        self._clear_2d_pose_preview("Capture to see 2D Pose")
        try:
            self._fig_3d = plt.figure(figsize=(4,3))
            self._ax_3d = self._fig_3d.add_subplot(111, projection='3d')
            self._canvas_3d = FigureCanvasTkAgg(self._fig_3d, master=right_vis_col)
            self._canvas_3d_widget = self._canvas_3d.get_tk_widget()
            self._canvas_3d_widget.pack(pady=1, fill=tk.BOTH, expand=True)
            self._clear_3d_plot("Initializing 3D Plot...")
        except Exception as e:
            rospy.logerr(f"Failed to initialize 3D plot: {e}")
            ttk.Label(right_vis_col, text="Error initializing 3D plot.").pack(pady=1, fill=tk.BOTH, expand=True)

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

        self._apply_topic_btn = ttk.Button(topic_select_frame, text="Apply & Restart Subs",
                                            command=self._on_apply_topic_change)
        self._apply_topic_btn.pack(side=tk.LEFT)


        checkbox_frame = ttk.Frame(controls_outer_frame)
        checkbox_frame.pack(fill=tk.X, pady=2)
        self._overlay_checkbutton = ttk.Checkbutton(checkbox_frame, text="Show Overlay", variable=self._show_overlay_var)
        self._overlay_checkbutton.pack(anchor=tk.W)
        self._filter_checkbutton = ttk.Checkbutton(checkbox_frame, text="Apply Median Filter", variable=self._apply_median_filter_var)
        self._filter_checkbutton.pack(anchor=tk.W)
        self._calib_checkbox = ttk.Checkbutton(checkbox_frame, text="Use Calibration File",
                                                variable=self._use_calibration_var,
                                                command=self._on_toggle_calibration)
        self._calib_checkbox.pack(anchor=tk.W)
        if not self._calibration_loaded_successfully: self._calib_checkbox.config(state=tk.DISABLED)
        else: self._calib_checkbox.config(state=tk.NORMAL)

        self._capture_btn = ttk.Button(controls_outer_frame, text="Capture Pose (W)", command=self._trigger_capture)
        self._capture_btn.pack(fill=tk.X, pady=(5,2))
        self.root.bind("<KeyPress-w>", lambda _: self._trigger_capture())
        self.root.bind("<KeyPress-W>", lambda _: self._trigger_capture())

        label_counts_frame = ttk.Labelframe(right_controls_and_counts_frame, text="Label Counts", padding="5")
        label_counts_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        for label_name in POSE_LABELS:
            lbl = ttk.Label(label_counts_frame, textvariable=self._label_counts_vars[label_name])
            lbl.pack(anchor=tk.W)

        status_lbl = ttk.Label(main_frame, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_lbl.pack(side=tk.BOTTOM, pady=(5,0), fill=tk.X)
        self._update_gui_status("Initializing GUI...")


    def _initialize_csv(self):
        self._csv_path = os.path.join(SAVE_ROOT_CSV, CSV_FILENAME)
        try:
            file_exists = os.path.exists(self._csv_path)
            csv_dir = os.path.dirname(self._csv_path)
            if not os.path.exists(csv_dir): os.makedirs(csv_dir)
            if csv_dir and not os.access(csv_dir, os.W_OK): raise IOError(f"No write permission: {csv_dir}")
            if file_exists and not os.access(self._csv_path, os.W_OK): raise IOError(f"No write permission: {self._csv_path}")

            with open(self._csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists or os.path.getsize(self._csv_path) == 0:
                    rospy.loginfo(f"Creating/initializing CSV: {self._csv_path}")
                    header = ['label', 'timestamp', 'rgb_source_topic', 'calibration_used',
                                'median_filter_applied', 'openpose_conf_threshold',
                                'num_2d_peaks_detected_raw', 'num_2d_peaks_above_conf',
                                'num_3d_points_initial', 'num_3d_points_final']
                    header.extend([f'{c}{i}_rel' for i in range(21) for c in ('x', 'y', 'z')])
                    writer.writerow(header)
                else: rospy.loginfo(f"Appending to existing CSV: {self._csv_path}")
        except IOError as e:
            rospy.logfatal(f"Cannot access/write CSV file: {self._csv_path}. Error: {e}")
            self._update_gui_status(f"Error: Cannot access CSV: {e}")
            self._set_controls_state_internal(tk.DISABLED)

    def _unregister_subscribers(self):
        """Pomoćna metoda za odjavu postojećih ROS subscribera."""
        if hasattr(self, 'ts') and self.ts is not None:
            self.ts.callbacks.clear()
            if hasattr(self, 'rgb_sub_filter') and self.rgb_sub_filter is not None:
                self.rgb_sub_filter.sub.unregister()
                self.rgb_sub_filter = None
            if hasattr(self, 'depth_sub_filter') and self.depth_sub_filter is not None:
                self.depth_sub_filter.sub.unregister()
                self.depth_sub_filter = None
            if hasattr(self, 'info_sub_filter') and self.info_sub_filter is not None:
                self.info_sub_filter.sub.unregister()
                self.info_sub_filter = None
            self.ts = None
        rospy.loginfo("ROS subscribers unregistered.")


    def _initialize_ros_subscriptions(self):
        """Inicijalizira ili ponovno inicijalizira ROS message filtere i subscriber-e."""
        self._unregister_subscribers()

        try:
            rospy.loginfo(f"Initializing ROS subscribers with RGB topic: {self.current_rgb_topic_to_subscribe}")
            self.rgb_sub_filter = message_filters.Subscriber(self.current_rgb_topic_to_subscribe, Image)
            self.depth_sub_filter = message_filters.Subscriber(REGISTERED_DEPTH_TOPIC, Image)
            self.info_sub_filter = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)

            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub_filter, self.depth_sub_filter, self.info_sub_filter],
                queue_size=10, slop=MESSAGE_FILTER_SLOP
            )
            self.ts.registerCallback(self._synchronized_callback)
            rospy.loginfo(f"Message filters synchronized for RGB: {self.current_rgb_topic_to_subscribe}")
            self._update_gui_status("Waiting for synchronized data...")
            self._set_controls_state_internal(tk.NORMAL)
        except Exception as e:
            rospy.logfatal(f"Failed to initialize message_filters: {e}")
            self._update_gui_status("FATAL: Error setting up ROS Sync!")
            self._set_controls_state_internal(tk.DISABLED)

    def _on_apply_topic_change(self):
        """Poziva se kada korisnik klikne gumb za primjenu promjene RGB topica."""
        selected_topic_key = self._rgb_topic_selection_var.get()
        new_topic_value = RGB_TOPIC_OPTIONS.get(selected_topic_key)

        if new_topic_value and new_topic_value != self.current_rgb_topic_to_subscribe:
            rospy.loginfo(f"RGB topic change requested from '{self.current_rgb_topic_to_subscribe}' to '{new_topic_value}'. Restarting subscribers.")
            self.current_rgb_topic_to_subscribe = new_topic_value
            self._update_gui_status(f"Applying RGB topic: {new_topic_value}. Restarting subscribers...")
            self._set_controls_state_internal(tk.DISABLED)

            self.root.update_idletasks()

            self._initialize_ros_subscriptions()

        elif not new_topic_value:
            rospy.logwarn(f"Invalid topic key selected: {selected_topic_key}")
            messagebox.showwarning("Topic Error", "Invalid RGB topic selected.", parent=self.root)
        else:
            rospy.loginfo("Selected RGB topic is the same as current. No change needed.")


    def _on_toggle_calibration(self):
        is_checked = self._use_calibration_var.get()
        if is_checked and not self._calibration_loaded_successfully:
            rospy.logwarn("Calibration checkbox was checked, but calibration file is not loaded. Forcing off.")
            self._use_calibration_var.set(False)
            is_checked = False
        new_desired_active_state = is_checked and self._calibration_loaded_successfully
        if new_desired_active_state != self._calibration_active:
            self._calibration_active = new_desired_active_state
            status_msg = "Using calibration file." if self._calibration_active else \
                            ("Calibration file not loaded. Using CameraInfo." if not self._calibration_loaded_successfully and is_checked else "Using CameraInfo topic.")
            rospy.loginfo(f"Calibration active state: {self._calibration_active}. Status: {status_msg}")
            self._update_gui_status(status_msg)


    def _synchronized_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self.is_shutting_down: return

        if not self._camera_info_topic_received:
            try:
                if len(info_msg.K) == 9:
                    fx, fy, cx, cy = info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5]
                    if fx > 0 and fy > 0 and cx > 0 and cy > 0:
                        self._fx_topic, self._fy_topic, self._cx_topic, self._cy_topic = fx, fy, cx, cy
                        self._camera_info_topic_received = True
                        rospy.loginfo(f"Camera Info from TOPIC received: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                        if not self._calibration_active:
                            self._gui_queue.put(GuiTask("update_status", "Ready to capture (using CameraInfo)."))
            except Exception as e: rospy.logerr(f"Error processing CameraInfo from TOPIC: {e}")

        try:
            cv_rgb_source = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")

            processed_rgb = cv_rgb_source.copy()

            if self.current_rgb_topic_to_subscribe == RGB_TOPIC_RAW and \
                self._calibration_active and \
                self._K_calib is not None and self._D_calib is not None:

                h_orig, w_orig = cv_rgb_source.shape[:2]
                if self._calib_img_width == w_orig and self._calib_img_height == h_orig:
                    rospy.logdebug("Applying undistortion to RAW source using K_calib, D_calib.")
                    processed_rgb = cv2.undistort(cv_rgb_source, self._K_calib, self._D_calib, None, self._K_calib)
                else:
                    rospy.logwarn_throttle(10, f"RAW source: Calibration image size ({self._calib_img_width}x{self._calib_img_height}) "
                                            f"differs from stream ({w_orig}x{h_orig}). Skipping undistort.")

            with self._data_lock:
                self._latest_synced_data["rgb"] = processed_rgb
                self._latest_synced_data["depth"] = cv_depth.copy()
                self._latest_synced_data["stamp"] = rgb_msg.header.stamp

            overlay_img = self._create_overlay_image(processed_rgb, cv_depth)
            self._gui_queue.put(GuiTask("update_main_preview", (processed_rgb, overlay_img)))
            self._gui_queue.put(GuiTask("update_depth_preview", cv_depth))

        except CvBridgeError as e: rospy.logerr_throttle(5, f"CvBridge error: {e}")
        except Exception as e: rospy.logerr_throttle(5, f"Error processing synced messages: {e}")

    def _update_main_preview(self, payload: Tuple[np.ndarray, Optional[np.ndarray]]):
        if self.is_shutting_down: return
        try:
            rgb_bgr, overlay_bgr = payload
            show_overlay = self._show_overlay_var.get()
            image_to_show = overlay_bgr if show_overlay and overlay_bgr is not None else rgb_bgr
            if image_to_show is not None:
                display_rgb = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)
                display_resized = self._resize_for_display(display_rgb)
                pil_img = PILImage.fromarray(display_resized)
                tk_img = ImageTk.PhotoImage(pil_img)
                if hasattr(self, '_main_view_label') and self._main_view_label.winfo_exists():
                    self._main_view_label.configure(image=tk_img, text='')
                    self._main_view_label.image = tk_img 
        except Exception as e: rospy.logerr(f"Error _update_main_preview: {e}")

    def _update_depth_preview(self, frame_depth_16uc1: np.ndarray):
        if self.is_shutting_down: return
        try:
            min_d, max_d_thresh = VALID_DEPTH_THRESHOLD_MM
            display_depth = frame_depth_16uc1.copy()
            display_depth[display_depth < min_d] = max_d_thresh
            display_depth[display_depth > max_d_thresh] = max_d_thresh
            norm_image = cv2.normalize(display_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colorized_depth = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
            colorized_depth_rgb = cv2.cvtColor(colorized_depth, cv2.COLOR_BGR2RGB)
            depth_display_resized = self._resize_for_display(colorized_depth_rgb)
            pil_img = PILImage.fromarray(depth_display_resized)
            tk_img = ImageTk.PhotoImage(pil_img)
            if hasattr(self, '_depth_label') and self._depth_label.winfo_exists():
                self._depth_label.configure(image=tk_img, text='')
                self._depth_label.image = tk_img
        except Exception as e: rospy.logerr(f"Error _update_depth_preview: {e}")

    def _clear_depth_preview(self, text=""):
        if self.is_shutting_down: return
        try:
            if hasattr(self, '_depth_label') and self._depth_label.winfo_exists():
                self._depth_label.configure(image=None, text=text) 
                self._depth_label.image = None 
        except Exception as e: rospy.logerr(f"Error _clear_depth_preview: {e}")

    def _update_2d_pose_preview(self, payload: Tuple[np.ndarray, Optional[List[bool]]]):
        if self.is_shutting_down: return
        try:
            frame_bgr, validity_mask = payload
            display_img = frame_bgr.copy()
            peaks_to_draw = self._pending_capture_data.peaks_2d_filtered

            if validity_mask is not None and peaks_to_draw is not None and not np.all(np.isnan(peaks_to_draw)):
                for i, (p1_idx, p2_idx) in enumerate(hand_connections):
                    if p1_idx < peaks_to_draw.shape[0] and p2_idx < peaks_to_draw.shape[0] and \
                        not np.isnan(peaks_to_draw[p1_idx]).any() and not np.isnan(peaks_to_draw[p2_idx]).any() and \
                        p1_idx < len(validity_mask) and p2_idx < len(validity_mask) and \
                        validity_mask[p1_idx] and validity_mask[p2_idx]:

                        p1 = tuple(peaks_to_draw[p1_idx, :2].astype(int))
                        p2 = tuple(peaks_to_draw[p2_idx, :2].astype(int))
                        h_img, w_img = display_img.shape[:2]
                        if 0 <= p1[0] < w_img and 0 <= p1[1] < h_img and \
                            0 <= p2[0] < w_img and 0 <= p2[1] < h_img:
                            cv2.line(display_img, p1, p2, limb_colors[i % len(limb_colors)], 2)
                for i in range(peaks_to_draw.shape[0]):
                    if i < len(validity_mask) and not np.isnan(peaks_to_draw[i]).any():
                        point = tuple(peaks_to_draw[i, :2].astype(int))
                        color = VALID_POINT_COLOR if validity_mask[i] else INVALID_POINT_COLOR
                        h_img, w_img = display_img.shape[:2]
                        if 0 <= point[0] < w_img and 0 <= point[1] < h_img:
                            cv2.circle(display_img, point, 4, color, thickness=-1)

            pose_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pose_rgb_resized = self._resize_for_display(pose_rgb)
            pil_img = PILImage.fromarray(pose_rgb_resized)
            tk_img = ImageTk.PhotoImage(pil_img)
            if hasattr(self, '_pose_2d_label') and self._pose_2d_label.winfo_exists():
                self._pose_2d_label.configure(image=tk_img, text='')
                self._pose_2d_label.image = tk_img 
        except Exception as e: rospy.logerr(f"Error _update_2d_pose_preview: {e}")


    def _clear_2d_pose_preview(self, text=""):
        if self.is_shutting_down: return
        try:
            if hasattr(self, '_pose_2d_label') and self._pose_2d_label.winfo_exists():
                self._pose_2d_label.configure(image=None, text=text) 
                self._pose_2d_label.image = None 
        except Exception as e: rospy.logerr(f"Error _clear_2d_pose_preview: {e}")

    def _update_3d_plot(self, points_3d_relative: Optional[np.ndarray], title: str =""):
        if self.is_shutting_down: return
        try:
            if not hasattr(self, '_canvas_3d') or not self._canvas_3d_widget.winfo_exists(): return
            self._ax_3d.clear()
            plot_x, plot_y, plot_z = [], [], []
            if points_3d_relative is not None:
                if not isinstance(points_3d_relative, np.ndarray): points_3d_relative = np.array(points_3d_relative, dtype=np.float32)

                valid_mask_for_plot = ~np.isnan(points_3d_relative).any(axis=1)
                valid_points_for_plot = points_3d_relative[valid_mask_for_plot]

                if valid_points_for_plot.shape[0] > 0:
                    plot_x, plot_y, plot_z = valid_points_for_plot[:, 0], -valid_points_for_plot[:, 1], -valid_points_for_plot[:, 2]
                    self._ax_3d.scatter(plot_x, plot_y, plot_z, c='r', marker='o', s=25, depthshade=True)

                    for s_idx, e_idx in hand_connections:
                        if s_idx < len(valid_mask_for_plot) and e_idx < len(valid_mask_for_plot) and \
                            valid_mask_for_plot[s_idx] and valid_mask_for_plot[e_idx]:
                            ps, pe = points_3d_relative[s_idx], points_3d_relative[e_idx]
                            self._ax_3d.plot([ps[0], pe[0]], [-ps[1], -pe[1]], [-ps[2], -pe[2]], 'b-')

            self._ax_3d.set_xlabel('X rel (m)'); self._ax_3d.set_ylabel('-Y rel (m)'); self._ax_3d.set_zlabel('-Z rel (m)')
            self._ax_3d.set_title(title)
            if len(plot_x) > 0:
                max_range_val = np.array([np.nanmax(plot_x)-np.nanmin(plot_x), np.nanmax(plot_y)-np.nanmin(plot_y), np.nanmax(plot_z)-np.nanmin(plot_z)]).max() / 2.0
                if max_range_val < 0.01 or np.isnan(max_range_val): max_range_val = 0.1
                mid_x = np.nanmean(plot_x); mid_y = np.nanmean(plot_y); mid_z = np.nanmean(plot_z)
                if np.isnan(mid_x): mid_x=0.0; mid_y=0.0; mid_z=0.0
                self._ax_3d.set_xlim(mid_x - max_range_val, mid_x + max_range_val); self._ax_3d.set_ylim(mid_y - max_range_val, mid_y + max_range_val); self._ax_3d.set_zlim(mid_z - max_range_val, mid_z + max_range_val)
            else: self._ax_3d.set_xlim([-0.2, 0.2]); self._ax_3d.set_ylim([-0.2, 0.2]); self._ax_3d.set_zlim([-0.3, 0.1])
            try: self._ax_3d.set_aspect('auto')
            except NotImplementedError: rospy.logwarn_once("Equal aspect ratio for 3D plot not supported by this Matplotlib backend.")
            self._canvas_3d.draw_idle()
        except Exception as e:
            rospy.logerr(f"Error updating 3D plot: {e}")
            self._clear_3d_plot("Error updating plot")

    def _clear_3d_plot(self, title=""):
        if self.is_shutting_down: return
        try:
            if hasattr(self, '_ax_3d') and self._canvas_3d_widget.winfo_exists():
                self._ax_3d.clear()
                self._ax_3d.set_xlabel('X (rel)'); self._ax_3d.set_ylabel('Y (rel)'); self._ax_3d.set_zlabel('Z (rel)')
                self._ax_3d.set_title(title)
                self._ax_3d.set_xlim([-0.2, 0.2]); self._ax_3d.set_ylim([-0.2, 0.2]); self._ax_3d.set_zlim([-0.1, 0.3])
                self._canvas_3d.draw_idle()
        except Exception as e: rospy.logerr(f"Error clearing 3D plot: {e}")

    def _update_gui_status(self, message: str):
        if hasattr(self, '_gui_queue'): self._gui_queue.put(GuiTask("update_status", message))

    def _set_controls_state(self, state: str):
        if hasattr(self, '_gui_queue'): self._gui_queue.put(GuiTask("set_controls_state", state))

    def _resize_for_display(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_h, max_w = DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH
        if h > max_h or w > max_w:
            scale = min(max_h / h, max_w / w)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image

    def _create_overlay_image(self, rgb_bgr: np.ndarray, depth_16uc1: np.ndarray) -> Optional[np.ndarray]:
        if rgb_bgr is None or depth_16uc1 is None: return None
        try:
            min_d, max_d_thresh = VALID_DEPTH_THRESHOLD_MM
            display_depth = depth_16uc1.copy()
            display_depth[display_depth < min_d] = min_d
            display_depth[display_depth > max_d_thresh] = max_d_thresh
            norm_image = cv2.normalize(display_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colorized_depth = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
            h_rgb, w_rgb = rgb_bgr.shape[:2]; h_depth, w_depth = colorized_depth.shape[:2]
            if h_rgb != h_depth or w_rgb != w_depth:
                colorized_depth_resized = cv2.resize(colorized_depth, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            else: colorized_depth_resized = colorized_depth
            return cv2.addWeighted(rgb_bgr, OVERLAY_ALPHA, colorized_depth_resized, 1.0 - OVERLAY_ALPHA, 0.0)
        except Exception as e:
            rospy.logerr(f"Error creating overlay image: {e}")
            return rgb_bgr

    def _get_depth_from_neighborhood(self, depth_map: np.ndarray, cx: float, cy: float, size: int =3) -> float:
        if depth_map is None: return np.nan
        if size % 2 == 0: size += 1
        radius = size // 2
        h, w = depth_map.shape; ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < w and 0 <= iy < h): return np.nan

        y_min, y_max = max(0, iy - radius), min(h, iy + radius + 1)
        x_min, x_max = max(0, ix - radius), min(w, ix + radius + 1)
        neighborhood = depth_map[y_min:y_max, x_min:x_max]
        min_d, max_d_thresh = VALID_DEPTH_THRESHOLD_MM
        valid_depths_mm = neighborhood[(neighborhood >= min_d) & (neighborhood <= max_d_thresh)]

        if valid_depths_mm.size < max(1, (size*size)//4): return np.nan

        std_dev_mm = np.std(valid_depths_mm)
        if std_dev_mm > DEPTH_STD_DEV_THRESHOLD_MM:
            rospy.logdebug(f"High std dev ({std_dev_mm:.1f}mm) for ({cx:.0f},{cy:.0f}) in depth neighborhood. NaN.")
            return np.nan
        return float(np.median(valid_depths_mm) / 1000.0)

    def _filter_3d_outliers(self, keypoints_3d: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
        if keypoints_3d is None or keypoints_3d.shape[0] != 21 :
            rospy.logwarn(f"[_filter_3d_outliers] Input keypoints_3d is None or not shape (21,3). Got: {keypoints_3d.shape if keypoints_3d is not None else 'None'}")
            return np.full((21, 3), np.nan, dtype=np.float32), [False] * 21

        points_filtered = keypoints_3d.copy()
        validity_mask = ~np.isnan(points_filtered).any(axis=1)

        if not validity_mask[0]:
            rospy.logwarn("Wrist keypoint (0) is invalid, cannot filter outliers relative to it.")
            return points_filtered, validity_mask.tolist()

        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        num_removed = 0
        for i in range(1, 21):
            if validity_mask[i]:
                dist_sq = np.sum(points_filtered[i]**2)
                if dist_sq > max_dist_sq:
                    points_filtered[i] = np.nan
                    validity_mask[i] = False
                    num_removed += 1
        if num_removed > 0:
            rospy.loginfo(f"Removed {num_removed} 3D outlier points (dist from wrist > {OUTLIER_XYZ_THRESHOLD_M:.2f}m).")

        return points_filtered, validity_mask.tolist()


    def _trigger_capture(self):
        if self.is_shutting_down: return
        if self._waiting_for_label:
            self._cancel_pending_save()
        self._set_controls_state_internal(tk.DISABLED)
        threading.Thread(target=self._capture_logic, daemon=True).start()

    def _capture_logic(self):
        current_metadata = PendingCaptureMetadata()
        current_metadata.timestamp = datetime.datetime.now().isoformat()
        current_metadata.rgb_source_topic = self.current_rgb_topic_to_subscribe
        current_metadata.calibration_used = self._calibration_active
        current_metadata.median_filter_applied = self._apply_median_filter_var.get()
        current_metadata.openpose_conf_threshold_value = OPENPOSE_CONFIDENCE_THRESHOLD

        rospy.loginfo(f"[_CAPTURE_LOGIC] Start. Calib active: {self._calibration_active}, RGB Topic: {self.current_rgb_topic_to_subscribe}")

        with self._data_lock:
            rgb_frame_for_pose = self._latest_synced_data["rgb"]
            depth_frame_mm = self._latest_synced_data["depth"]

        current_fx: Optional[float]=None; current_fy: Optional[float]=None
        current_cx: Optional[float]=None; current_cy: Optional[float]=None
        intrinsics_source = "NONE"

        if self._calibration_active and self._K_calib is not None:
            current_fx, current_fy = self._K_calib[0,0], self._K_calib[1,1]
            current_cx, current_cy = self._K_calib[0,2], self._K_calib[1,2]
            intrinsics_source = "CALIBRATION FILE"
        elif self._camera_info_topic_received and self._fx_topic is not None:
            current_fx, current_fy = self._fx_topic, self._fy_topic
            current_cx, current_cy = self._cx_topic, self._cy_topic
            intrinsics_source = "CAMERA_INFO TOPIC"

        if current_fx is None:
            rospy.logwarn("[_CAPTURE_LOGIC] Capture aborted: No camera intrinsics available.")
            self._gui_queue.put(GuiTask("update_status", "Error: No intrinsics for 3D!"))
            self._gui_queue.put(GuiTask("enable_controls")); return

        if rgb_frame_for_pose is None or depth_frame_mm is None:
            rospy.logwarn("[_CAPTURE_LOGIC] Capture aborted: Missing RGB or Depth frame.")
            self._gui_queue.put(GuiTask("update_status", "Cannot capture: Missing image data."))
            self._gui_queue.put(GuiTask("enable_controls")); return

        self._gui_queue.put(GuiTask("update_status", f"Detecting hand (intrinsics: {intrinsics_source})..."))

        depth_image_to_process = depth_frame_mm
        filter_applied_msg = "Raw"
        if current_metadata.median_filter_applied:
            try:
                ksize = MEDIAN_FILTER_KERNEL_SIZE
                if ksize % 2 == 0: ksize += 1
                depth_image_to_process = cv2.medianBlur(depth_frame_mm, ksize)
                filter_applied_msg = f"Median(k={ksize})"
            except Exception as e:
                rospy.logerr(f"[_CAPTURE_LOGIC] Error applying Median filter: {e}. Using raw.")
                filter_applied_msg = "Raw(Filter Failed)"

        all_peaks_2d_output = None
        try:
            all_peaks_2d_output = self._hand_estimator(rgb_frame_for_pose)
            if all_peaks_2d_output is None:
                rospy.logwarn("[_CAPTURE_LOGIC] _hand_estimator returned None.")
            else:
                rospy.loginfo(f"[_CAPTURE_LOGIC] _hand_estimator returned array of shape: {all_peaks_2d_output.shape if hasattr(all_peaks_2d_output, 'shape') else 'N/A'}")
        except Exception as e:
            rospy.logerr(f"[_CAPTURE_LOGIC] Error during hand estimation: {e}")
            self._gui_queue.put(GuiTask("update_status", "Hand Estimation Error!"))
            self._gui_queue.put(GuiTask("enable_controls")); return

        if all_peaks_2d_output is None or not isinstance(all_peaks_2d_output, np.ndarray):
            rospy.logwarn(f"[_CAPTURE_LOGIC] OpenPose did not return a NumPy array. Returned: {type(all_peaks_2d_output)}")
            self._gui_queue.put(GuiTask("update_status", "Hand not detected or invalid OpenPose output type."))
            self._gui_queue.put(GuiTask("enable_controls")); return

        if all_peaks_2d_output.ndim != 2:
            rospy.logwarn(f"[_CAPTURE_LOGIC] OpenPose output is not a 2D array. ndim: {all_peaks_2d_output.ndim}")
            self._gui_queue.put(GuiTask("update_status", "Invalid OpenPose output dimensions."))
            self._gui_queue.put(GuiTask("enable_controls")); return

        current_metadata.num_2d_peaks_detected_raw = all_peaks_2d_output.shape[0]

        has_confidence_scores = all_peaks_2d_output.shape[1] >= 3

        peaks_2d_filtered_coords_list = []
        original_indices_passed_conf_filter = []

        self._pending_capture_data.peaks_2d_filtered = np.full((21,2), np.nan, dtype=np.float32)

        if not has_confidence_scores:
            rospy.logwarn_throttle(10, "[_CAPTURE_LOGIC] OpenPose output does not include confidence scores (expected shape (N,3), got shape (N,2)). Assuming all detected points are valid.")
            current_metadata.openpose_conf_threshold_value = -1.0
            for i in range(current_metadata.num_2d_peaks_detected_raw):
                original_keypoint_index = i
                if original_keypoint_index < 21:
                    coords = all_peaks_2d_output[i, :2]
                    self._pending_capture_data.peaks_2d_filtered[original_keypoint_index] = coords
                    peaks_2d_filtered_coords_list.append(coords)
                    original_indices_passed_conf_filter.append(original_keypoint_index)
            current_metadata.num_2d_peaks_above_conf = current_metadata.num_2d_peaks_detected_raw
        else:
            for i in range(current_metadata.num_2d_peaks_detected_raw):
                original_keypoint_index = i
                if original_keypoint_index < 21:
                    if all_peaks_2d_output[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD:
                        coords = all_peaks_2d_output[i, :2]
                        self._pending_capture_data.peaks_2d_filtered[original_keypoint_index] = coords
                        peaks_2d_filtered_coords_list.append(coords)
                        original_indices_passed_conf_filter.append(original_keypoint_index)
            current_metadata.num_2d_peaks_above_conf = len(peaks_2d_filtered_coords_list)

        peaks_2d_for_projection = np.array(peaks_2d_filtered_coords_list, dtype=np.float32) if peaks_2d_filtered_coords_list else np.empty((0,2), dtype=np.float32)

        if current_metadata.num_2d_peaks_above_conf == 0 :
            status_msg = f"No 2D keypoints meet criteria. Raw: {current_metadata.num_2d_peaks_detected_raw}. Press W."
            if has_confidence_scores:
                status_msg = f"No 2D keypoints above confidence ({OPENPOSE_CONFIDENCE_THRESHOLD}). Raw: {current_metadata.num_2d_peaks_detected_raw}. Press W."
            rospy.logwarn(f"[_CAPTURE_LOGIC] {status_msg}")
            self._gui_queue.put(GuiTask("update_pose_preview", (rgb_frame_for_pose.copy(), [False]*21 )))
            self._gui_queue.put(GuiTask("clear_3d", "No reliable 2D keypoints"))
            self._gui_queue.put(GuiTask("update_status", status_msg))
            self._gui_queue.put(GuiTask("enable_controls")); return

        status_msg_3d = f"Detected {current_metadata.num_2d_peaks_above_conf} 2D peaks. Projecting ({filter_applied_msg}, Intr: {intrinsics_source})..."
        self._gui_queue.put(GuiTask("update_status", status_msg_3d)); rospy.loginfo(f"[_CAPTURE_LOGIC] {status_msg_3d}")

        keypoints_cam_3d_raw_projection = np.full((current_metadata.num_2d_peaks_above_conf, 3), np.nan, dtype=np.float32)
        validity_mask_3d_initial_projection = [False] * current_metadata.num_2d_peaks_above_conf

        for i in range(current_metadata.num_2d_peaks_above_conf):
            x_pix, y_pix = float(peaks_2d_for_projection[i, 0]), float(peaks_2d_for_projection[i, 1])
            z_m = self._get_depth_from_neighborhood(depth_image_to_process, x_pix, y_pix, size=DEPTH_NEIGHBORHOOD_SIZE)
            if not np.isnan(z_m):
                if current_fx is not None and current_fy is not None and current_cx is not None and current_cy is not None:
                    x_cam = (x_pix - current_cx) * z_m / current_fx
                    y_cam = (y_pix - current_cy) * z_m / current_fy
                    keypoints_cam_3d_raw_projection[i] = [x_cam, y_cam, z_m]
                    validity_mask_3d_initial_projection[i] = True

        current_metadata.num_3d_points_initial = np.sum(validity_mask_3d_initial_projection)
        rospy.loginfo(f"[_CAPTURE_LOGIC] Initial 3D points from projection: {current_metadata.num_3d_points_initial}/{current_metadata.num_2d_peaks_above_conf}")

        keypoints_cam_3d_full = np.full((21,3), np.nan, dtype=np.float32)
        for i, original_idx in enumerate(original_indices_passed_conf_filter):
            if original_idx < 21 and validity_mask_3d_initial_projection[i]:
                keypoints_cam_3d_full[original_idx] = keypoints_cam_3d_raw_projection[i]

        keypoints_rel_3d = np.full_like(keypoints_cam_3d_full, np.nan)
        if not np.isnan(keypoints_cam_3d_full[0]).any():
            wrist_3d_cam = keypoints_cam_3d_full[0].copy()
            keypoints_rel_3d = keypoints_cam_3d_full - wrist_3d_cam
            keypoints_rel_3d[0] = [0.0, 0.0, 0.0]
        else:
            rospy.logwarn("[_CAPTURE_LOGIC] Wrist (idx 0) has no valid 3D projection for relative coords.")
            keypoints_rel_3d = keypoints_cam_3d_full

        keypoints_rel_3d_filtered, validity_mask_final = self._filter_3d_outliers(keypoints_rel_3d)
        num_3d_points_final = np.sum(validity_mask_final)

        self._pending_capture_data.capture_metadata = current_metadata
        self._pending_capture_data.validity_mask_final = validity_mask_final

        self._gui_queue.put(GuiTask("update_pose_preview", (rgb_frame_for_pose.copy(), validity_mask_final)))
        plot_title = f"Relative 3D ({num_3d_points_final} valid, Intr: {intrinsics_source.split(' ')[0]})"
        self._gui_queue.put(GuiTask("update_3d", (keypoints_rel_3d_filtered, plot_title)))

        if num_3d_points_final < MIN_VALID_KEYPOINTS_FOR_SAVE:
            status_msg_skip = f"Only {num_3d_points_final} valid points after filter (min {MIN_VALID_KEYPOINTS_FOR_SAVE}). Press W."
            rospy.logwarn(f"[_CAPTURE_LOGIC] {status_msg_skip}")
            self._gui_queue.put(GuiTask("update_status", status_msg_skip))
            self._gui_queue.put(GuiTask("enable_controls")); return

        if not validity_mask_final[0]:
            rospy.logwarn("[_CAPTURE_LOGIC] Wrist keypoint (idx 0) became invalid after outlier filter.")
            self._gui_queue.put(GuiTask("update_status", "Invalid wrist after filter! Press W."))
            self._gui_queue.put(GuiTask("enable_controls")); return

        self._pending_capture_data.keypoints_rel_3d = keypoints_rel_3d_filtered.copy()
        self._waiting_for_label = True

        label_keys_parts = []
        for i, label_text in enumerate(POSE_LABELS):
            key_char = str(i + 1) if i < 9 else ("O" if label_text.lower() == "other" and POSE_LABELS.index("other") == i else str(i+1) )
            label_keys_parts.append(f"{key_char}={label_text}")

        status_msg_label = f"{num_3d_points_final} valid 3D pts. Press key: {', '.join(label_keys_parts)} or Q to Cancel"
        self._gui_queue.put(GuiTask("update_status", status_msg_label))

    def _handle_label_key(self, label_index: int):
        if not self._waiting_for_label: return
        if 0 <= label_index < len(POSE_LABELS):
            label = POSE_LABELS[label_index]
            rospy.loginfo(f"Label key pressed. Saving as '{label}'...")
            self._save_pending_data(label)
            self._label_counts[label] += 1
            self._gui_queue.put(GuiTask("update_label_counts", None))
        else: rospy.logwarn(f"Invalid label index received: {label_index}")
        self._waiting_for_label = False
        self._pending_capture_data = CaptureData()
        self._gui_queue.put(GuiTask("enable_controls"))

    def _handle_cancel_key(self, event=None): 
        if not self._waiting_for_label: return
        rospy.loginfo("Cancel key 'Q' pressed. Discarding capture.")
        self._cancel_pending_save()

    def _cancel_pending_save(self):
        self._waiting_for_label = False
        self._pending_capture_data = CaptureData()
        self._gui_queue.put(GuiTask("update_status", "Capture cancelled. Ready."))
        self._gui_queue.put(GuiTask("clear_3d", "Capture Cancelled"))
        self._gui_queue.put(GuiTask("clear_2d", "Capture Cancelled"))
        self._gui_queue.put(GuiTask("enable_controls"))

    def _save_pending_data(self, label: str):
        data_to_save = self._pending_capture_data
        points_rel_3d = data_to_save.keypoints_rel_3d
        metadata = data_to_save.capture_metadata

        if points_rel_3d is None or metadata is None:
            rospy.logerr("No keypoints or metadata available to save!")
            self._update_gui_status("Error: No data to save!")
            return

        num_3d_points_final = np.sum(data_to_save.validity_mask_final) if data_to_save.validity_mask_final is not None else 0
        self._update_gui_status(f"Saving {num_3d_points_final} valid points as '{label}'...")

        try:
            with open(self._csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                row_data = [
                    label, metadata.timestamp, metadata.rgb_source_topic, metadata.calibration_used,
                    metadata.median_filter_applied, metadata.openpose_conf_threshold_value,
                    metadata.num_2d_peaks_detected_raw, metadata.num_2d_peaks_above_conf,
                    metadata.num_3d_points_initial, num_3d_points_final
                ]
                flat_coords = []
                for i in range(21):
                    if data_to_save.validity_mask_final and \
                        i < len(data_to_save.validity_mask_final) and \
                        data_to_save.validity_mask_final[i] and \
                        i < points_rel_3d.shape[0] and \
                        not np.isnan(points_rel_3d[i]).any():
                        flat_coords.extend([f"{coord:.6f}" for coord in points_rel_3d[i]])
                    else:
                        flat_coords.extend(["", "", ""])
                row_data.extend(flat_coords)
                writer.writerow(row_data)
            self._update_gui_status(f"Saved pose '{label}' ({num_3d_points_final} valid). Ready.")
            rospy.loginfo(f"Pose '{label}' data appended to {self._csv_path}")
        except IOError as e:
            rospy.logerr(f"Error writing CSV: {e}")
            self._update_gui_status(f"Error writing CSV: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error saving CSV: {e}")
            self._update_gui_status(f"Error writing CSV! Check logs.")

    def run(self):
        try:
            rospy.loginfo("Starting GUI main loop...")
            self.root.mainloop()
        except Exception as e: rospy.logerr(f"Error in GUI main loop: {e}")
        finally:
            rospy.loginfo("Exiting GUI main loop.")
            self.on_close()

    def ros_spin_target(self):
        try:
            rospy.spin()
            rospy.loginfo("ROS spin finished.")
        except rospy.ROSInterruptException: rospy.loginfo("ROS spin interrupted.")
        except Exception as e: rospy.logerr(f"Error in ROS spin thread: {e}")

    def on_close(self):
        if not self.is_shutting_down:
            rospy.loginfo("Close button pressed. Shutting down...")
            self.is_shutting_down = True
            if self._after_id:
                try: self.root.after_cancel(self._after_id)
                except tk.TclError: pass
                self._after_id = None

            self._unregister_subscribers()

            if not rospy.is_shutdown(): rospy.signal_shutdown("GUI closed by user")
            try:
                if hasattr(self, '_fig_3d') and plt.fignum_exists(self._fig_3d.number): 
                    plt.close(self._fig_3d) 
            except Exception as e: rospy.logwarn(f"Error closing matplotlib figure: {e}")
            try:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.quit(); self.root.destroy()
            except tk.TclError: pass
            except Exception as e: rospy.logwarn(f"Error destroying Tk root: {e}")
            rospy.loginfo("GUI resources released.")

    def _pump_gui_queue(self):
        if self.is_shutting_down: return
        task = None
        try:
            while not self._gui_queue.empty():
                task = self._gui_queue.get_nowait()
                cmd, payload = task.command, task.payload
                if not hasattr(self, 'root') or not self.root.winfo_exists(): break

                if cmd == "update_status":
                    if hasattr(self, '_status_var'): self._status_var.set(str(payload))
                elif cmd == "update_main_preview":
                    if payload is not None: self._update_main_preview(payload)
                elif cmd == "update_depth_preview":
                    if payload is not None: self._update_depth_preview(payload)
                elif cmd == "clear_depth_preview":
                    self._clear_depth_preview(str(payload) if payload is not None else "")
                elif cmd == "update_pose_preview":
                    if payload is not None: self._update_2d_pose_preview(payload)
                elif cmd == "clear_2d":
                    self._clear_2d_pose_preview(str(payload) if payload is not None else "")
                elif cmd == "update_3d":
                    if payload is not None and len(payload) == 2: self._update_3d_plot(payload[0], str(payload[1]))
                elif cmd == "clear_3d":
                    self._clear_3d_plot(str(payload) if payload is not None else "")
                elif cmd == "enable_controls":
                    self._set_controls_state_internal(tk.NORMAL)
                elif cmd == "set_controls_state":
                    self._set_controls_state_internal(payload if payload in [tk.NORMAL, tk.DISABLED] else tk.NORMAL)
                elif cmd == "update_label_counts":
                    self._update_label_counts_display()
                else:
                    rospy.logwarn(f"Unknown GUI command: {cmd}")
                self._gui_queue.task_done()
        except queue.Empty: pass
        except Exception as e:
            command_name = task.command if task else "Unknown"
            rospy.logerr(f"Error pumping GUI queue (Command: {command_name}): {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        finally:
            if not self.is_shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
                self._after_id = self.root.after(20, self._pump_gui_queue)

    def _set_controls_state_internal(self, target_state: str):
        if hasattr(self, '_capture_btn'): self._capture_btn.config(state=target_state)
        if hasattr(self, '_filter_checkbutton'): self._filter_checkbutton.config(state=target_state)
        if hasattr(self, '_overlay_checkbutton'): self._overlay_checkbutton.config(state=target_state)
        if hasattr(self, '_calib_checkbox'):
            self._calib_checkbox.config(state=target_state if self._calibration_loaded_successfully else tk.DISABLED)

        if hasattr(self, '_rgb_topic_dropdown'): self._rgb_topic_dropdown.config(state="readonly")
        if hasattr(self, '_apply_topic_btn'): self._apply_topic_btn.config(state=tk.NORMAL)


    def _update_label_counts_display(self):
        for label, count_var in self._label_counts_vars.items():
            count_var.set(f"{label}: {self._label_counts[label]}")


if __name__ == "__main__":
    collector = None
    try:
        collector = DepthHandCollector()
        collector.run()
    except rospy.ROSInterruptException: rospy.loginfo("ROS node interrupted by ROS master.")
    except KeyboardInterrupt: rospy.loginfo("Script interrupted by user (Ctrl+C).")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if collector and hasattr(collector, 'on_close') and not collector.is_shutting_down:
            rospy.loginfo("Performing final cleanup in main finally block...")
            collector.on_close()
        if not rospy.is_shutdown():
            rospy.loginfo("ROS is not shutdown in main finally, signaling shutdown now.")
            rospy.signal_shutdown("Script main block finished or error.")
        rospy.loginfo("DepthHandCollector script finished.")
        print("\nExiting DepthHandCollector.")