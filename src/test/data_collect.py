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

        self.sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)

        self.current_frame = None
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            rospy.loginfo(f"[HandPoseCollector] Torch device: {dev_name}")
        else:
            rospy.loginfo("[HandPoseCollector] Torch device: CPU")

        self.root = tk.Tk()
        self.root.title("Hand Pose Collector")

        self.capture_button = ttk.Button(self.root, text="Take picture", command=self.on_capture_click)
        self.capture_button.pack(pady=10)

        self.csv_path = os.path.join(os.getcwd(), "data", "hand_keypoints.csv")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                cols = ["label"]
                for i in range(21):
                    cols.append(f"x{i}")
                    cols.append(f"y{i}")
                header_line = ",".join(cols)
                f.write(header_line + "\n")

        self.spin_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.spin_thread.start()

    def ros_spin(self):
        rospy.spin()

    def image_callback(self, ros_image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        self.current_frame = cv_img
        cv2.imshow("Live Kinect color", cv_img)
        cv2.waitKey(1)

    def on_capture_click(self):
        if self.current_frame is None:
            self.info_label.config(text="No frame captured yet!")
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
        cmb = ttk.Combobox(frame_label, textvariable=pose_var, 
                           values=POSE_LABELS, state="readonly")
        cmb.pack(side="left", padx=5)

        frame_buttons = ttk.Frame(popup)
        frame_buttons.pack(pady=10)

        def on_save():
            label_str = pose_var.get()
            # center on wrist
            wrist_x, wrist_y = peaks[0,0], peaks[0,1]
            norm_points = []
            for i in range(21):
                x = peaks[i,0] - wrist_x
                y = peaks[i,1] - wrist_y
                norm_points.append((x,y))

            line_elems = [label_str]
            for (xx,yy) in norm_points:
                line_elems.append(f"{xx:.2f}")
                line_elems.append(f"{yy:.2f}")
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