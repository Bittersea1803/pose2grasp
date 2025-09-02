#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import queue
import threading
import csv
import datetime
from dataclasses import dataclass, field

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage
from PIL import ImageTk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPENPOSE_PATH = os.path.join(PROJECT_ROOT, "pytorch-openpose")
sys.path.append(OPENPOSE_PATH)
from src.hand import Hand

DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
CSV_PATH = os.path.join(DATA_DIR, "collected_hand_poses.csv")
os.makedirs(DATA_DIR, exist_ok=True)

RGB_TOPIC = "/camera/rgb/image_rect_color"
DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
INFO_TOPIC = "/camera/rgb/camera_info"


VALID_DEPTH_MIN_MM = 400
VALID_DEPTH_MAX_MM = 1500
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
MEDIAN_FILTER_KERNEL_SIZE = 3

OPENPOSE_CONFIDENCE_THRESHOLD = 0.2  
MIN_VALID_KEYPOINTS_FOR_SAVE = 18
OUTLIER_XYZ_THRESHOLD_M = 0.25
MAX_LIMB_LENGTH_M = 0.10

POSE_LABELS = ["basic", "wide", "pinch", "scissor"]

HAND_CONNECTIONS = [
    [0,1], [1,2], [2,3], [3,4], # Thumb
    [0,5], [5,6], [6,7], [7,8], # Index
    [0,9], [9,10], [10,11], [11,12], # Middle
    [0,13], [13,14], [14,15], [15,16], # Ring
    [0,17], [17,18], [18,19], [19,20] # Pinky
]

LIMB_COLORS = [
    [255,0,0], [255,85,0], [255,170,0], [255,255,0],
    [170,255,0], [85,255,0], [0,255,0], [0,255,85],
    [0,255,170], [0,255,255], [0,170,255], [0,85,255],
    [0,0,255], [85,0,255], [170,0,255], [255,0,255],
    [255,0,170], [255,0,85]
]

@dataclass
class CaptureData:
    keypoints_3d: np.ndarray = None # 21x3 array of 3D coordinates
    peaks_2d: np.ndarray = None # 21x3 array (x, y, confidence)
    validity: list = None # List of 21 booleans
    metadata: dict = field(default_factory=dict)

class HandCollector:    
    def __init__(self):
        rospy.init_node("hand_collector", anonymous=True)
        rospy.loginfo("Starting")
        self.bridge = CvBridge()
        self.shutdown = False
        self.gui_queue = queue.Queue()
        self.data_lock = threading.Lock()
        
        # Data buffers
        self.latest_data = {
            "rgb": None,
            "depth": None
        }
        
        # Capture state management
        self.pending = CaptureData()
        self.waiting_label = False
        
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        model_path = os.path.join(OPENPOSE_PATH, "model", "hand_pose_model.pth")
        if not os.path.exists(model_path):
            rospy.logfatal(f"OpenPose model not found at: {model_path}")
            sys.exit(1)
        self.hand_est = Hand(model_path)
        rospy.loginfo("OpenPose model loaded successfully")
        
        self.root = tk.Tk()
        self.root.title("Hand Pose Data Collection")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        # GUI variables
        self.status = tk.StringVar(value="Initializing")
        self.median_var = tk.BooleanVar(value=True)
        self.overlay_var = tk.BooleanVar(value=True)
        
        self.setup_gui()
        self.load_counts()
        self.init_csv()
        self.setup_ros()
        self.bind_keys()
        
        ros_thread = threading.Thread(target=rospy.spin, daemon=True)
        ros_thread.start()
        
        self.process_queue()

    def setup_gui(self):

        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        
        # Left side
        display_frame = ttk.Frame(main)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # RGB camera feed
        self.rgb_lbl = ttk.Label(display_frame)
        self.rgb_lbl.grid(row=0, column=0, padx=2, pady=2)
        
        # Depth camera feed
        self.depth_lbl = ttk.Label(display_frame)
        self.depth_lbl.grid(row=1, column=0, padx=2, pady=2)
        
        # Captured pose with skeleton overlay
        self.pose_lbl = ttk.Label(display_frame)
        self.pose_lbl.grid(row=0, column=1, padx=2, pady=2)
        
        # 3D visualization plot
        self.fig_3d = plt.figure(figsize=(4, 3))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=display_frame)
        self.canvas_3d.get_tk_widget().grid(row=1, column=1, padx=2, pady=2)
        
        # Right side
        control_frame = ttk.LabelFrame(main, text="Controls", padding=5)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Checkboxes for options
        overlay_check = ttk.Checkbutton(control_frame, text="Overlay", variable=self.overlay_var)
        overlay_check.pack()
        
        median_check = ttk.Checkbutton(control_frame, text="Median Filter", variable=self.median_var)
        median_check.pack()
        
        # Capture button
        self.cap_btn = ttk.Button(control_frame, text="Capture (W)", command=self.capture)
        self.cap_btn.pack(pady=5)
        
        # Sample count
        self.count_lbls = {}
        for label in POSE_LABELS:
            self.count_lbls[label] = tk.StringVar(value=f"{label}: 0")
            count_label = ttk.Label(control_frame, textvariable=self.count_lbls[label])
            count_label.pack()
        
        # Status bar
        status_bar = ttk.Label(main, textvariable=self.status, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def load_counts(self):        
        # Initialize count dictionary
        self.counts = {}
        for label in POSE_LABELS:
            self.counts[label] = 0
        
        # Count existing samples if CSV exists
        if os.path.exists(CSV_PATH):
            try:
                with open(CSV_PATH, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row.get('label')
                        if label is not None and label in self.counts:
                            self.counts[label] = self.counts[label] + 1
            except Exception as e:
                rospy.logwarn(f"Could not load existing counts: {e}")
        
        self.update_counts()

    def init_csv(self):        
        # Check if CSV exists and has content
        csv_exists = os.path.exists(CSV_PATH)
        csv_empty = True
        
        if csv_exists:
            csv_empty = os.path.getsize(CSV_PATH) == 0
        
        # Create CSV with headers if needed
        if not csv_exists or csv_empty:
            with open(CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                
                header = []
                
                # Metadata columns
                header.append('label')
                header.append('timestamp')
                header.append('rgb_source_topic')
                header.append('calibration_used')
                header.append('median_filter_applied')
                header.append('openpose_conf_threshold')
                header.append('num_2d_peaks_detected_raw')
                header.append('num_2d_peaks_above_conf')
                header.append('num_3d_points_initial')
                header.append('num_3d_points_final')
                
                # 3D coordinate columns (21 keypoints × 3 coordinates)
                for i in range(21):
                    header.append(f'x{i}_rel')
                    header.append(f'y{i}_rel')
                    header.append(f'z{i}_rel') 
                
                writer.writerow(header)
                rospy.loginfo(f"Created CSV file with {len(header)} columns")

    def setup_ros(self):
        
        def callback(rgb_msg, depth_msg, info_msg):
            if self.shutdown:
                return
            
            # Extract camera intrinsic parameters on first message
            if self.fx is None:
                if hasattr(info_msg, 'K') and len(info_msg.K) >= 6:
                    if info_msg.K[0] > 0:
                        self.fx = info_msg.K[0]
                        self.fy = info_msg.K[4]
                        self.cx = info_msg.K[2]
                        self.cy = info_msg.K[5]
                        rospy.loginfo(f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}")
            
            try:
                # Convert ROS messages to OpenCV format
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                
                # Store latest synchronized data
                with self.data_lock:
                    self.latest_data["rgb"] = rgb_image
                    self.latest_data["depth"] = depth_image
                
                # Trigger GUI update
                self.gui_queue.put(("preview", None))
                
            except Exception as e:
                rospy.logwarn(f"Message conversion error: {e}")
        
        # Create individual subscribers
        subscribers = []
        subscribers.append(message_filters.Subscriber(RGB_TOPIC, Image))
        subscribers.append(message_filters.Subscriber(DEPTH_TOPIC, Image))
        subscribers.append(message_filters.Subscriber(INFO_TOPIC, CameraInfo))
        
        # Create approximate time synchronizer
        # Queue size of 10, time tolerance of 0.1 seconds
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=10, slop=0.1
        )
        self.sync.registerCallback(callback)
        
        rospy.loginfo("ROS subscribers configured and synchronized")

    def bind_keys(self):        
        # Number keys for quick labeling
        def handle_key_1(event):
            self.save_with_label("basic")
        def handle_key_2(event):
            self.save_with_label("wide")
        def handle_key_3(event):
            self.save_with_label("pinch")
        def handle_key_4(event):
            self.save_with_label("scissor")
        
        self.root.bind("<KeyPress-1>", handle_key_1)
        self.root.bind("<KeyPress-2>", handle_key_2)
        self.root.bind("<KeyPress-3>", handle_key_3)
        self.root.bind("<KeyPress-4>", handle_key_4)
        
        # W key for capture
        def handle_capture(event):
            self.capture()
        self.root.bind("<w>", handle_capture)
        self.root.bind("<W>", handle_capture)
        
        # Q key for cancel
        def handle_cancel(event):
            self.cancel()
        self.root.bind("<q>", handle_cancel)
        self.root.bind("<Q>", handle_cancel)

    def get_depth(self, depth_map, u, v):
        # Boundary check
        if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
            return np.nan
        
        # Define neighborhood region
        r = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_start = max(0, v - r)
        y_end = min(depth_map.shape[0], v + r + 1)
        x_start = max(0, u - r)
        x_end = min(depth_map.shape[1], u + r + 1)
        
        # Extract neighborhood
        neighborhood = depth_map[y_start:y_end, x_start:x_end]
        
        # Filter valid depth values using vectorized operations
        valid_mask = (neighborhood >= VALID_DEPTH_MIN_MM) & \
                     (neighborhood <= VALID_DEPTH_MAX_MM)
        valid = neighborhood[valid_mask]
        
        # Check if we have enough valid samples
        min_valid_count = max(1, (DEPTH_NEIGHBORHOOD_SIZE ** 2) // 4)
        if valid.size < min_valid_count:
            return np.nan
        
        # Check depth stability (low standard deviation indicates reliable reading)
        if np.std(valid) > DEPTH_STD_DEV_THRESHOLD_MM:
            return np.nan
        
        # Return median depth converted to meters
        return float(np.median(valid)) / 1000.0

    def filter_outliers(self, pts_3d_rel):
        filtered = pts_3d_rel.copy()
        valid = []
        
        # Check each keypoint for NaN values
        for i in range(21):
            point_is_valid = True
            
            # Check if any coordinate is NaN
            if np.isnan(filtered[i][0]) or np.isnan(filtered[i][1]) or np.isnan(filtered[i][2]):
                point_is_valid = False
            
            valid.append(point_is_valid)
        
        # If wrist is invalid, can't filter by distance
        if not valid[0]:
            return filtered, valid
        
        # Check distance from wrist for all other keypoints
        max_dist_squared = OUTLIER_XYZ_THRESHOLD_M * OUTLIER_XYZ_THRESHOLD_M
        
        for i in range(1, 21):
            if valid[i]:
                # Calculate squared Euclidean distance from wrist
                dist_squared = filtered[i][0]**2 + filtered[i][1]**2 + filtered[i][2]**2
                
                if dist_squared > max_dist_squared:
                    # Mark as invalid if too far
                    filtered[i] = np.array([np.nan, np.nan, np.nan])
                    valid[i] = False
        
        return filtered, valid

    def filter_limb_length(self, pts_3d_rel, valid_mask):
        pts = pts_3d_rel.copy()
        mask = list(valid_mask)
        
        # Iterate filtering up to 3 times to handle cascading removals
        for iteration in range(3):
            removed_count = 0
            
            # Check each bone connection
            for connection in HAND_CONNECTIONS:
                p1 = connection[0]
                p2 = connection[1]
                
                if mask[p1] and mask[p2]:
                    # Calculate bone length
                    diff = pts[p1] - pts[p2]
                    limb_length_squared = diff[0]**2 + diff[1]**2 + diff[2]**2
                    
                    max_length_squared = MAX_LIMB_LENGTH_M * MAX_LIMB_LENGTH_M
                    
                    if limb_length_squared > max_length_squared:
                        # Remove the point further from wrist
                        dist_p1_squared = pts[p1][0]**2 + pts[p1][1]**2 + pts[p1][2]**2
                        dist_p2_squared = pts[p2][0]**2 + pts[p2][1]**2 + pts[p2][2]**2
                        
                        if dist_p1_squared > dist_p2_squared:
                            idx_to_remove = p1
                        else:
                            idx_to_remove = p2
                        
                        if mask[idx_to_remove]:
                            pts[idx_to_remove] = np.array([np.nan, np.nan, np.nan])
                            mask[idx_to_remove] = False
                            removed_count = removed_count + 1
            
            # Stop if no points were removed in this iteration
            if removed_count == 0:
                break
        
        return pts, mask

    def capture(self):        
        # Don't capture if already waiting for label
        if self.waiting_label:
            return
        
        # Get latest synchronized data
        with self.data_lock:
            rgb = self.latest_data["rgb"]
            depth = self.latest_data["depth"]
        
        # Check prerequisites
        if rgb is None:
            self.status.set("Waiting for camera feed...")
            return
        
        if self.fx is None:
            self.status.set("Waiting for camera calibration...")
            return
        
        self.status.set("Processing hand detection...")
        
        # Apply median filter to depth if enabled (reduces noise)
        if self.median_var.get():
            depth = cv2.medianBlur(depth, MEDIAN_FILTER_KERNEL_SIZE)
        
        # Run OpenPose hand detection
        peaks = self.hand_est(rgb)
        
        # Check if hand was detected
        if peaks is None or peaks.shape[0] < 21:
            self.status.set("No hand detected")
            return
        
        # Initialize 3D point array
        pts_3d = np.full((21, 3), np.nan)
        valid = []
        
        # Convert 2D detections to 3D using depth
        for i in range(21):
            point_is_valid = False
            
            if i < peaks.shape[0]:
                # Get confidence score
                if peaks.shape[1] > 2:
                    confidence = peaks[i, 2]
                else:
                    confidence = 1.0
                
                # Only process high-confidence detections
                if confidence > OPENPOSE_CONFIDENCE_THRESHOLD:
                    # Get 2D pixel coordinates
                    u = int(peaks[i, 0])
                    v = int(peaks[i, 1])
                    
                    # Get depth value at this pixel
                    z = self.get_depth(depth, u, v)
                    
                    if not np.isnan(z):
                        # Back-project to 3D using pinhole camera model
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        pts_3d[i] = np.array([x, y, z])
                        point_is_valid = True
            
            valid.append(point_is_valid)
        
        # Store initial valid count for statistics
        num_3d_initial = sum(valid)
        
        # Normalize coordinates relative to wrist (keypoint 0)
        if valid[0]:
            wrist_pos = pts_3d[0].copy()
            for i in range(21):
                pts_3d[i] = pts_3d[i] - wrist_pos
            pts_3d[0] = np.array([0.0, 0.0, 0.0])
        
        # Apply filtering pipeline
        pts_3d, valid = self.filter_outliers(pts_3d)
        pts_3d, valid = self.filter_limb_length(pts_3d, valid)
        
        # Count final valid points  
        n_valid = sum(valid)
        if n_valid < MIN_VALID_KEYPOINTS_FOR_SAVE or not valid[0]:
            self.status.set(f"Too few points: {n_valid}")
            return

        # Store successful capture
        self.pending = CaptureData(
            keypoints_3d=pts_3d,
            peaks_2d=peaks[:21],
            validity=valid,
            metadata={
                "timestamp": datetime.datetime.now().isoformat(),
                "median": self.median_var.get(),
                "num_3d_initial": num_3d_initial
            }
        )
        
        # Update visualizations
        self.gui_queue.put(("show_capture", (rgb, self.pending)))
        self.gui_queue.put(("plot_3d", pts_3d.astype(np.float32)))
        
        # Wait for user to assign label
        self.waiting_label = True
        self.status.set(f"Captured {n_valid} points. Press 1-4 for label, Q to cancel")

    def save_with_label(self, label):        
        # Check if we have pending capture
        if not self.waiting_label:
            return
        
        if self.pending.keypoints_3d is None:
            return
        
        # Calculate statistics for metadata
        peaks = self.pending.peaks_2d
        if peaks is not None:
            num_2d_raw = peaks.shape[0]
        else:
            num_2d_raw = 0
        
        # Count 2D detections above confidence threshold
        num_2d_conf = 0
        for i in range(min(21, num_2d_raw)):
            if i < peaks.shape[0]:
                if peaks.shape[1] > 2:
                    confidence = peaks[i, 2]
                else:
                    confidence = 1.0
                
                if confidence > OPENPOSE_CONFIDENCE_THRESHOLD:
                    num_2d_conf = num_2d_conf + 1
        
        # Count final valid 3D points
        num_3d_final = sum(self.pending.validity)
        
        # Build CSV row
        row = []
        
        # Add metadata columns
        row.append(label)
        row.append(self.pending.metadata.get("timestamp"))
        row.append(RGB_TOPIC)
        row.append(False)  # calibration_used
        row.append(self.pending.metadata.get("median"))
        row.append(OPENPOSE_CONFIDENCE_THRESHOLD)
        row.append(num_2d_raw)
        row.append(num_2d_conf)
        row.append(self.pending.metadata.get("num_3d_initial", num_3d_final))
        row.append(num_3d_final)
        
        # Add 3D coordinate data
        for i in range(21):
            if self.pending.validity[i]:
                # Save coordinates with 6 decimal precision
                row.append(f"{self.pending.keypoints_3d[i][0]:.6f}")
                row.append(f"{self.pending.keypoints_3d[i][1]:.6f}")
                row.append(f"{self.pending.keypoints_3d[i][2]:.6f}")
            else:
                # Empty values for invalid keypoints
                row.append("")
                row.append("")
                row.append("")
        
        # Write to CSV file
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Update statistics
        self.counts[label] = self.counts[label] + 1
        self.update_counts()
        
        # Reset capture state
        self.waiting_label = False
        self.status.set(f"Saved as '{label}'. Ready for next capture (W)")
        
        rospy.loginfo(f"Saved capture with label '{label}', {num_3d_final} valid points")

    def cancel(self):
        if self.waiting_label:
            self.waiting_label = False
            self.status.set("Capture cancelled. Ready for next (W)")

    def update_counts(self):
        for label in POSE_LABELS:
            count_text = f"{label}: {self.counts[label]}"
            self.count_lbls[label].set(count_text)

    def process_queue(self):
        if self.shutdown:
            return
        
        try:
            # Process all pending GUI updates
            while not self.gui_queue.empty():
                cmd, data = self.gui_queue.get_nowait()
                
                if cmd == "preview":
                    # Update live camera feeds
                    with self.data_lock:
                        rgb = self.latest_data["rgb"]
                        depth = self.latest_data["depth"]
                    
                    if rgb is not None:
                        # Create display image based on overlay setting
                        if not self.overlay_var.get() or depth is None:
                            display_img = rgb
                        else:
                            # Create RGB-D overlay visualization
                            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                            display_img = cv2.addWeighted(rgb, 0.5, depth_colored, 0.5, 0)
                        
                        self.show_img(self.rgb_lbl, display_img)
                    
                    if depth is not None:
                        # Show colorized depth map
                        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        self.show_img(self.depth_lbl, depth_colored)
                
                elif cmd == "show_capture":
                    # Display captured frame with skeleton overlay
                    rgb, cap = data
                    img = rgb.copy()
                    
                    # Draw bone connections
                    for i, connection in enumerate(HAND_CONNECTIONS):
                        p1 = connection[0]
                        p2 = connection[1]
                        
                        if cap.validity[p1] and cap.validity[p2]:
                            pt1 = (int(cap.peaks_2d[p1, 0]), int(cap.peaks_2d[p1, 1]))
                            pt2 = (int(cap.peaks_2d[p2, 0]), int(cap.peaks_2d[p2, 1]))
                            
                            color_idx = i % len(LIMB_COLORS)
                            color = LIMB_COLORS[color_idx]
                            
                            cv2.line(img, pt1, pt2, color, 2)
                    
                    # Draw keypoints
                    for i in range(min(21, cap.peaks_2d.shape[0])):
                        pt = (int(cap.peaks_2d[i, 0]), int(cap.peaks_2d[i, 1]))
                        
                        if cap.validity[i]:
                            color = (0, 255, 0)  # Green for valid
                        else:
                            color = (0, 0, 255)  # Red for invalid
                        
                        cv2.circle(img, pt, 3, color, -1)
                    
                    self.show_img(self.pose_lbl, img)
                
                elif cmd == "plot_3d":
                    # Update 3D visualization
                    self.ax_3d.clear()

                    # Convert to float32 for plotting
                    arr = np.array(data, dtype=np.float32)

                    # Valid only of none is NaN
                    valid = ~np.isnan(arr).any(axis=1)

                    if np.any(valid):
                        pts = arr[valid]

                        # Plot keypoints
                        self.ax_3d.scatter(pts[:, 0], -pts[:, 1], -pts[:, 2],
                                        c='r', marker='o', s=50)

                        # Plot bone connections (s istom maskom)
                        for p1, p2 in HAND_CONNECTIONS:
                            if p1 < len(valid) and p2 < len(valid) and valid[p1] and valid[p2]:
                                self.ax_3d.plot(
                                    [arr[p1, 0], arr[p2, 0]],
                                    [-arr[p1, 1], -arr[p2, 1]],
                                    [-arr[p1, 2], -arr[p2, 2]],
                                    'b-', linewidth=1
                                )

                    # Set consistent view limits
                    self.ax_3d.set_xlim([-0.15, 0.15])
                    self.ax_3d.set_ylim([-0.15, 0.15])
                    self.ax_3d.set_zlim([-0.15, 0.15])

                    # Axis labels
                    self.ax_3d.set_xlabel('X')
                    self.ax_3d.set_ylabel('-Y')
                    self.ax_3d.set_zlabel('-Z')

                    # Equal aspect ratio
                    try:
                        self.ax_3d.set_box_aspect([1, 1, 1])
                    except AttributeError:
                        pass

                    self.canvas_3d.draw()

                    
        except Exception as e:
            rospy.logwarn(f"GUI update error: {e}")
        
        # Schedule next update (50ms = 20 FPS)
        self.root.after(50, self.process_queue)

    def show_img(self, label_widget, img):
        if img is None:
            return
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for GUI display
        img_resized = cv2.resize(img_rgb, (320, 240))
        
        # Convert to PIL format
        pil_img = PILImage.fromarray(img_resized)
        
        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(pil_img)
        
        # Update label widget
        label_widget.configure(image=photo)
        label_widget.image = photo  # Keep reference to prevent garbage collection

    def close(self):
        self.shutdown = True
        
        # Shutdown ROS if still running
        if not rospy.is_shutdown():
            rospy.signal_shutdown("User closed GUI")
        
        # Close matplotlib figure
        try:
            plt.close(self.fig_3d)
        except Exception:
            pass
        
        # Destroy GUI
        self.root.destroy()

    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


if __name__ == "__main__":
    try:
        collector = HandCollector()
        collector.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except KeyboardInterrupt:
        rospy.loginfo("User interrupted Ctrl+C")
