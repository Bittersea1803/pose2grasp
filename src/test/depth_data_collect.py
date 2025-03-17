import rospy
import cv2
import numpy as np
import copy
import torch
import os
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image as PILImage
from PIL import ImageTk

pytorch_openpose_path = "/home/openpose_user/src/pose2grasp/src/pytorch-openpose"
sys.path.append(pytorch_openpose_path)

from src.hand import Hand
from src import util

POSE_LABELS = ["basic", "wide", "pinch", "scissor"]

class HandPoseCollector:
    def __init__(self):
        rospy.init_node("hand_pose_collector_node", anonymous=True)
        self.bridge = CvBridge()

        hand_model_path = os.path.join(pytorch_openpose_path, "model", "hand_pose_model.pth")
        self.hand_estimator = Hand(hand_model_path)

        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        self.current_frame = None
        self.current_depth = None

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            rospy.loginfo(f"[HandPoseCollector] Torch device: {dev_name}")
        else:
            rospy.loginfo("[HandPoseCollector] Torch device: CPU")

        self.root = tk.Tk()
        self.root.title("Hand Pose Collector")

        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side="top", padx=10, pady=10)

        self.rgb_label = ttk.Label(self.image_frame, text="Wait for RGB Image")
        self.rgb_label.grid(row=0, column=0, padx=10, pady=10)

        self.depth_label = ttk.Label(self.image_frame, text="Wait for Depth Image")
        self.depth_label.grid(row=0, column=1, padx=10, pady=10)

        self.info_label = ttk.Label(self.root, text="Press 'Take picture' to capture hand pose.")
        self.info_label.pack(pady=5)

        self.capture_button = ttk.Button(self.root, text="Take picture", command=self.on_capture_click)
        self.capture_button.pack(pady=10)

        self.csv_path = os.path.join(os.getcwd(), "data", "depth_hand_keypoints.csv")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                cols = ["label"]
                for i in range(21):
                    cols.extend([f"x{i}", f"y{i}", f"z{i}"])
                header_line = ",".join(cols)
                f.write(header_line + "\n")

        self.update_display()

        self.spin_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.spin_thread.start()

    def ros_spin(self):
        rospy.spin()

    def image_callback(self, ros_image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error (color): %s", e)
            return

        self.current_frame = cv_img

    def depth_callback(self, ros_depth_image):
        try:
            # 16UC1 passthrough 32FC1
            cv_depth = self.bridge.imgmsg_to_cv2(ros_depth_image, desired_encoding="16UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error (depth): %s", e)
            return

        self.current_depth = cv_depth

    def get_depth_at_point(self, x, y):
        if self.current_depth is None:
            return 0
        h, w = self.current_depth.shape
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            return self.current_depth[iy, ix]
        return 0

    def update_display(self):
        if self.current_frame is not None:
            rgb_img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            pil_rgb = PILImage.fromarray(rgb_img)
            tk_rgb = ImageTk.PhotoImage(pil_rgb)
            self.rgb_label.configure(image=tk_rgb)
            self.rgb_label.image = tk_rgb

        if self.current_depth is not None:
            depth_norm = cv2.normalize(self.current_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            pil_depth = PILImage.fromarray(colored_depth)
            tk_depth = ImageTk.PhotoImage(pil_depth)
            self.depth_label.configure(image=tk_depth)
            self.depth_label.image = tk_depth

        self.root.after(30, self.update_display)

    def on_capture_click(self):
        if self.current_frame is None or self.current_depth is None:
            self.info_label.config(text="Waiting for both RGB and Depth frames!")
            return

        with torch.no_grad():
            peaks = self.hand_estimator(self.current_frame)

        if peaks is None or len(peaks) < 21:
            self.info_label.config(text="Hand not detected!")
            return

        canvas = copy.deepcopy(self.current_frame)
        canvas = util.draw_handpose(canvas, [peaks])
        self.show_preview_popup(canvas, peaks)

    def show_preview_popup(self, pose_img, peaks):
        popup = tk.Toplevel(self.root)
        popup.title("Preview & Label")

        pose_img_rgb = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(pose_img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        img_label = ttk.Label(popup, image=tk_img)
        img_label.image = tk_img
        img_label.pack()

        frame_label = ttk.Frame(popup)
        frame_label.pack(pady=5)

        lbl_txt = ttk.Label(frame_label, text="Label:")
        lbl_txt.pack(side="left", padx=5)

        pose_var = tk.StringVar(value=POSE_LABELS[0])
        cmb = ttk.Combobox(frame_label, textvariable=pose_var, values=POSE_LABELS, state="readonly")
        cmb.pack(side="left", padx=5)

        frame_buttons = ttk.Frame(popup)
        frame_buttons.pack(pady=10)

        def on_save():
            label_str = pose_var.get()
            wrist_x, wrist_y = peaks[0, 0], peaks[0, 1]
            norm_points = []
            for i in range(21):
                x = peaks[i, 0] - wrist_x
                y = peaks[i, 1] - wrist_y
                depth_val = self.get_depth_at_point(peaks[i, 0], peaks[i, 1])
                norm_points.append((x, y, depth_val))
            line_elems = [label_str]
            for (xx, yy, zz) in norm_points:
                line_elems.extend([f"{xx:.2f}", f"{yy:.2f}", f"{zz:.2f}"])
            line_to_write = ",".join(line_elems)
            with open(self.csv_path, "a") as f:
                f.write(line_to_write + "\n")
            self.info_label.config(text=f"Pose '{label_str}' saved to CSV!")
            popup.destroy()

        def on_cancel():
            self.info_label.config(text="Pose not saved.")
            popup.destroy()

        btn_save = ttk.Button(frame_buttons, text="Save", command=on_save)
        btn_save.pack(side="left", padx=10)
        btn_cancel = ttk.Button(frame_buttons, text="Cancel", command=on_cancel)
        btn_cancel.pack(side="left", padx=10)

    def run(self):
        self.root.mainloop()
        cv2.destroyAllWindows()
        rospy.signal_shutdown("GUI closed")

if __name__ == "__main__":
    collector = HandPoseCollector()
    collector.run()