import os, sys, queue, threading, csv, datetime
from dataclasses import dataclass, field
import cv2, numpy as np, rospy, torch
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # REQUIRED for 3D projection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPENPOSE_PATH = os.path.join(PROJECT_ROOT, "pytorch-openpose")
sys.path.append(OPENPOSE_PATH)
from src.hand import Hand

DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
CSV_PATH = os.path.join(DATA_DIR, "collected_hand_poses.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# Topics
RGB_TOPIC = "/camera/rgb/image_rect_color"
DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
INFO_TOPIC = "/camera/rgb/camera_info"

# Parameters
VALID_DEPTH_THRESHOLD_MM = (400, 1500)
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2
MIN_VALID_KEYPOINTS_FOR_SAVE = 18
OUTLIER_XYZ_THRESHOLD_M = 0.25
MAX_LIMB_LENGTH_M = 0.10
DEPTH_NEIGHBORHOOD_SIZE = 3
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
MEDIAN_FILTER_KERNEL_SIZE = 3

POSE_LABELS = ["basic", "wide", "pinch", "scissor"]
HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
LIMB_COLORS = [[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],[0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],[0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85]]

@dataclass
class CaptureData:
    keypoints_3d: np.ndarray = None
    peaks_2d: np.ndarray = None
    validity: list = None
    metadata: dict = field(default_factory=dict)

class HandCollector:
    def __init__(self):
        rospy.init_node("hand_collector", anonymous=True)
        rospy.loginfo("Starting Hand Collector...")
        self.bridge = CvBridge()
        self.shutdown = False
        self.gui_queue = queue.Queue()
        self.data_lock = threading.Lock()
        self.latest_data = {"rgb": None, "depth": None}
        self.pending = CaptureData()
        self.waiting_label = False
        self.fx = self.fy = self.cx = self.cy = None
        
        # OpenPose
        model_path = os.path.join(OPENPOSE_PATH, "model", "hand_pose_model.pth")
        if not os.path.exists(model_path):
            rospy.logfatal(f"Model not found: {model_path}")
            sys.exit(1)
        self.hand_est = Hand(model_path)
        
        # GUI
        self.root = tk.Tk()
        self.root.title("Hand Pose Collector")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        self.status = tk.StringVar(value="Initializing...")
        self.median_var = tk.BooleanVar(value=True)
        self.overlay_var = tk.BooleanVar(value=True)
        
        self.setup_gui()
        self.load_counts()
        self.init_csv()
        self.setup_ros()
        self.bind_keys()
        
        threading.Thread(target=rospy.spin, daemon=True).start()
        self.process_queue()

    def setup_gui(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        
        # Display area
        disp = ttk.Frame(main)
        disp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.rgb_lbl = ttk.Label(disp)
        self.rgb_lbl.grid(row=0, column=0, padx=2, pady=2)
        self.depth_lbl = ttk.Label(disp)
        self.depth_lbl.grid(row=1, column=0, padx=2, pady=2)
        self.pose_lbl = ttk.Label(disp)
        self.pose_lbl.grid(row=0, column=1, padx=2, pady=2)
        
        # 3D plot
        self.fig_3d = plt.figure(figsize=(4, 3))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=disp)
        self.canvas_3d.get_tk_widget().grid(row=1, column=1, padx=2, pady=2)
        
        # Controls
        ctrl = ttk.LabelFrame(main, text="Controls", padding=5)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        ttk.Checkbutton(ctrl, text="Overlay", variable=self.overlay_var).pack()
        ttk.Checkbutton(ctrl, text="Median Filter", variable=self.median_var).pack()
        self.cap_btn = ttk.Button(ctrl, text="Capture (W)", command=self.capture)
        self.cap_btn.pack(pady=5)
        
        # Counts
        self.count_lbls = {}
        for lbl in POSE_LABELS:
            self.count_lbls[lbl] = tk.StringVar(value=f"{lbl}: 0")
            ttk.Label(ctrl, textvariable=self.count_lbls[lbl]).pack()
        
        ttk.Label(main, textvariable=self.status, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def load_counts(self):
        self.counts = {l: 0 for l in POSE_LABELS}
        if os.path.exists(CSV_PATH):
            try:
                with open(CSV_PATH, 'r') as f:
                    for row in csv.DictReader(f):
                        if lbl := row.get('label'):
                            self.counts[lbl] = self.counts.get(lbl, 0) + 1
            except: pass
        self.update_counts()

    def init_csv(self):
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            with open(CSV_PATH, 'w', newline='') as f:
                hdr = ['label', 'timestamp', 'rgb_source_topic', 'calibration_used', 'median_filter_applied', 
                       'openpose_conf_threshold', 'num_2d_peaks_detected_raw', 'num_2d_peaks_above_conf', 
                       'num_3d_points_initial', 'num_3d_points_final']
                hdr.extend([f'{c}{i}_rel' for i in range(21) for c in ('x','y','z')])
                csv.writer(f).writerow(hdr)

    def setup_ros(self):
        def cb(rgb, depth, info):
            if self.shutdown: return
            if not self.fx and getattr(info, 'K', None) and len(info.K) >= 6 and info.K[0] > 0:
                self.fx, self.fy, self.cx, self.cy = info.K[0], info.K[4], info.K[2], info.K[5]
            try:
                with self.data_lock:
                    self.latest_data = {"rgb": self.bridge.imgmsg_to_cv2(rgb, "bgr8"), 
                                      "depth": self.bridge.imgmsg_to_cv2(depth, "16UC1")}
                self.gui_queue.put(("preview", None))
            except Exception as e: rospy.logwarn(f"Sync error: {e}")
        
        subs = [message_filters.Subscriber(t, Image) for t in [RGB_TOPIC, DEPTH_TOPIC]]
        subs.append(message_filters.Subscriber(INFO_TOPIC, CameraInfo))
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.1)
        self.sync.registerCallback(cb)

    def bind_keys(self):
        for i, lbl in enumerate(POSE_LABELS):
            self.root.bind(f"<KeyPress-{i+1}>", lambda e, l=lbl: self.save_with_label(l))
        self.root.bind("<w>", lambda e: self.capture())
        self.root.bind("<W>", lambda e: self.capture())
        self.root.bind("<q>", lambda e: self.cancel())
        self.root.bind("<Q>", lambda e: self.cancel())

    def get_depth(self, depth_map, u, v):
        if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]): return np.nan
        r = DEPTH_NEIGHBORHOOD_SIZE // 2
        neighborhood = depth_map[max(0,v-r):min(depth_map.shape[0],v+r+1), 
                                max(0,u-r):min(depth_map.shape[1],u+r+1)]
        valid = neighborhood[(neighborhood >= VALID_DEPTH_THRESHOLD_MM[0]) & 
                            (neighborhood <= VALID_DEPTH_THRESHOLD_MM[1])]
        if valid.size < max(1, (DEPTH_NEIGHBORHOOD_SIZE**2)//4) or np.std(valid) > DEPTH_STD_DEV_THRESHOLD_MM:
            return np.nan
        return float(np.median(valid)) / 1000.0

    def filter_outliers(self, pts_3d_rel):
        filtered, valid = pts_3d_rel.copy(), ~np.isnan(pts_3d_rel).any(axis=1)
        if not valid[0]: return filtered, valid.tolist()
        max_dist_sq = OUTLIER_XYZ_THRESHOLD_M ** 2
        for i in range(1, 21):
            if valid[i] and np.sum(filtered[i]**2) > max_dist_sq:
                filtered[i] = np.nan
                valid[i] = False
        return filtered, valid.tolist()

    def filter_limb_length(self, pts_3d_rel, valid):
        pts, mask = pts_3d_rel.copy(), list(valid)
        for _ in range(3):
            removed = 0
            for p1, p2 in HAND_CONNECTIONS:
                if mask[p1] and mask[p2] and np.sum((pts[p1] - pts[p2])**2) > MAX_LIMB_LENGTH_M**2:
                    idx = p1 if np.sum(pts[p1]**2) > np.sum(pts[p2]**2) else p2
                    if mask[idx]: pts[idx] = np.nan; mask[idx] = False; removed += 1
            if removed == 0: break
        return pts, mask

    def capture(self):
        if self.waiting_label: return
        with self.data_lock:
            rgb, depth = self.latest_data["rgb"], self.latest_data["depth"]
        if rgb is None or not self.fx:
            self.status.set("Waiting for camera & intrinsics..."); return
        
        self.status.set("Processing...")
        depth = cv2.medianBlur(depth, MEDIAN_FILTER_KERNEL_SIZE) if self.median_var.get() else depth
        
        peaks = self.hand_est(rgb)
        if peaks is None or peaks.shape[0] < 21:
            self.status.set("No hand detected"); return
        
        # Build 3D points
        pts_3d, valid = np.full((21, 3), np.nan), []
        for i in range(21):
            if i < peaks.shape[0] and (peaks[i, 2] if peaks.shape[1] > 2 else 1) > OPENPOSE_CONFIDENCE_THRESHOLD:
                u, v = int(peaks[i, 0]), int(peaks[i, 1])
                if not np.isnan(z := self.get_depth(depth, u, v)):
                    pts_3d[i] = [(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z]
                    valid.append(True)
                    continue
            valid.append(False)
        
        num_3d_initial = sum(valid)
        if valid[0]:
            pts_3d -= pts_3d[0]; pts_3d[0] = [0, 0, 0]
        
        pts_3d, valid = self.filter_outliers(pts_3d)
        pts_3d, valid = self.filter_limb_length(pts_3d, valid)
        
        if (n_valid := sum(valid)) < MIN_VALID_KEYPOINTS_FOR_SAVE or not valid[0]:
            self.status.set(f"Too few points: {n_valid}"); return
        
        self.pending = CaptureData(pts_3d, peaks[:21], valid, 
                                  {"timestamp": datetime.datetime.now().isoformat(),
                                   "median": self.median_var.get(),
                                   "num_3d_initial": num_3d_initial})
        self.gui_queue.put(("show_capture", (rgb, self.pending)))
        self.gui_queue.put(("plot_3d", pts_3d.astype(np.float32)))
        self.waiting_label = True
        self.status.set(f"Captured {n_valid} pts. Press 1-4 for label, Q to cancel")

    def save_with_label(self, label):
        if not self.waiting_label or self.pending.keypoints_3d is None: return
        
        peaks = self.pending.peaks_2d
        num_2d_raw = peaks.shape[0] if peaks is not None else 0
        num_2d_conf = sum(1 for i in range(min(21, num_2d_raw)) if i < peaks.shape[0] and 
                         (peaks[i, 2] if peaks.shape[1] > 2 else 1) > OPENPOSE_CONFIDENCE_THRESHOLD)
        num_3d_final = sum(self.pending.validity)
        
        row = [label, self.pending.metadata.get("timestamp"), RGB_TOPIC, False, 
               self.pending.metadata.get("median"), OPENPOSE_CONFIDENCE_THRESHOLD,
               num_2d_raw, num_2d_conf, self.pending.metadata.get("num_3d_initial", num_3d_final), num_3d_final]
        for i in range(21):
            row.extend([f"{x:.6f}" for x in self.pending.keypoints_3d[i]] if self.pending.validity[i] else ["", "", ""])
        
        with open(CSV_PATH, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        self.counts[label] += 1
        self.update_counts()
        self.waiting_label = False
        self.status.set("Saved. Ready for next capture (W)")

    def cancel(self):
        if self.waiting_label:
            self.waiting_label = False
            self.status.set("Cancelled. Ready for next capture (W)")

    def update_counts(self):
        for lbl in POSE_LABELS:
            self.count_lbls[lbl].set(f"{lbl}: {self.counts[lbl]}")

    def process_queue(self):
        if self.shutdown: return
        try:
            while not self.gui_queue.empty():
                cmd, data = self.gui_queue.get_nowait()
                if cmd == "preview":
                    with self.data_lock:
                        rgb, depth = self.latest_data["rgb"], self.latest_data["depth"]
                    if rgb is not None:
                        img = rgb if not self.overlay_var.get() or depth is None else cv2.addWeighted(
                            rgb, 0.5, cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET), 0.5, 0)
                        self.show_img(self.rgb_lbl, img)
                    if depth is not None:
                        self.show_img(self.depth_lbl, cv2.applyColorMap(
                            cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET))
                    
                elif cmd == "show_capture":
                    rgb, cap = data
                    img = rgb.copy()
                    for i, (p1, p2) in enumerate(HAND_CONNECTIONS):
                        if cap.validity[p1] and cap.validity[p2]:
                            cv2.line(img, tuple(cap.peaks_2d[p1,:2].astype(int)), 
                                    tuple(cap.peaks_2d[p2,:2].astype(int)), LIMB_COLORS[i%len(LIMB_COLORS)], 2)
                    for i in range(min(21, cap.peaks_2d.shape[0])):
                        cv2.circle(img, tuple(cap.peaks_2d[i,:2].astype(int)), 3, 
                                  (0,255,0) if cap.validity[i] else (0,0,255), -1)
                    self.show_img(self.pose_lbl, img)
                    
                elif cmd == "plot_3d":
                    self.ax_3d.clear()
                    data = np.array(data, dtype=np.float32)
                    valid = ~np.isnan(data).any(axis=1)
                    if (pts := data[valid]).size > 0:
                        self.ax_3d.scatter(pts[:,0], -pts[:,1], -pts[:,2], c='r', marker='o', s=50)
                        for p1, p2 in HAND_CONNECTIONS:
                            if p1 < len(valid) and p2 < len(valid) and valid[p1] and valid[p2]:
                                self.ax_3d.plot([data[p1,0], data[p2,0]], [-data[p1,1], -data[p2,1]], 
                                              [-data[p1,2], -data[p2,2]], 'b-', linewidth=1)
                        self.ax_3d.set_xlim([-0.15, 0.15]); self.ax_3d.set_ylim([-0.15, 0.15]); self.ax_3d.set_zlim([-0.15, 0.15])
                    self.ax_3d.set_xlabel('X'); self.ax_3d.set_ylabel('-Y'); self.ax_3d.set_zlabel('-Z')
                    try: self.ax_3d.set_box_aspect([1,1,1])
                    except: pass
                    self.canvas_3d.draw()
        except: pass
        self.root.after(50, self.process_queue)

    def show_img(self, lbl, img):
        if img is None: return
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 240))
        photo = ImageTk.PhotoImage(PILImage.fromarray(img))
        lbl.configure(image=photo)
        lbl.image = photo

    def close(self):
        self.shutdown = True
        if not rospy.is_shutdown(): rospy.signal_shutdown("GUI closed")
        try: plt.close(self.fig_3d)
        except: pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        collector = HandCollector()
        collector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupted")
    except KeyboardInterrupt:
        rospy.loginfo("User interrupted")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()