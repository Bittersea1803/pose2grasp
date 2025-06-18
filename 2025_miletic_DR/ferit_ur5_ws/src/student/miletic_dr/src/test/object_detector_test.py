#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#works in lab

import os
import sys
import threading
import numpy as np
import cv2
import rospy
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image as PILImage, ImageTk
from tf.transformations import quaternion_from_matrix


# CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
# CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
# CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"

OBJECT_POSE_TOPIC = "/plane_object_detector/biggest_object_position"


class ObjectDetectorLogic:
    def __init__(self, camera_intrinsics):
        self.fx = camera_intrinsics.K[0]
        self.fy = camera_intrinsics.K[4]
        self.cx = camera_intrinsics.K[2]
        self.cy = camera_intrinsics.K[5]
        
        self.VOXEL_SIZE = 0.005
        self.PLANE_DIST_THRESHOLD = 0.015
        self.DBSCAN_EPS = 0.025
        self.DBSCAN_MIN_POINTS = 50

    def create_pcd_from_rgbd_manual(self, rgb_image, depth_image):
        depth_16u = np.asarray(depth_image)
        rgb_bgr = np.asarray(rgb_image)
        H, W = depth_16u.shape
        valid_mask = (depth_16u > 400) & (depth_16u < 2000)
        indices = np.array(np.nonzero(valid_mask)).T
        if indices.shape[0] == 0:
            return o3d.geometry.PointCloud()

        us = indices[:, 1].astype(np.float32)
        vs = indices[:, 0].astype(np.float32)
        zs = depth_16u[indices[:, 0], indices[:, 1]].astype(np.float32) / 1000.0
        xs = (us - self.cx) * zs / self.fx
        ys = (vs - self.cy) * zs / self.fy
        points = np.stack((xs, ys, zs), axis=-1)
        colors_bgr = rgb_bgr[indices[:, 0], indices[:, 1]]
        colors_rgb = colors_bgr[:, ::-1] / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        return pcd

    def find_objects(self, pcd):
        if not pcd.has_points():
            return None, "Point cloud is empty", []

        pcd_down = pcd.voxel_down_sample(self.VOXEL_SIZE)
        if not pcd_down.has_points():
            return None, "Downsampled point cloud is empty", []

        try:
            plane_model, inliers = pcd_down.segment_plane(
                distance_threshold=self.PLANE_DIST_THRESHOLD, ransac_n=3, num_iterations=1000)
        except RuntimeError:
            return None, "Not enough points for plane segmentation", [pcd_down]

        plane_cloud = pcd_down.select_by_index(inliers)
        objects_cloud = pcd_down.select_by_index(inliers, invert=True)
        plane_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        geometries_to_draw = [plane_cloud]
        
        if not objects_cloud.has_points():
            return None, "No points left after removing plane", geometries_to_draw

        labels = np.array(objects_cloud.cluster_dbscan(
            eps=self.DBSCAN_EPS, min_points=self.DBSCAN_MIN_POINTS, print_progress=False))
        
        max_label = labels.max()
        if max_label < 0: return None, "DBSCAN found no clusters", geometries_to_draw

        valid_clusters = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > self.DBSCAN_MIN_POINTS:
                cluster_cloud = objects_cloud.select_by_index(cluster_indices)
                cluster_cloud.paint_uniform_color(np.random.rand(3))
                valid_clusters.append(cluster_cloud)
        
        if not valid_clusters: return None, "No valid clusters found", geometries_to_draw

        valid_clusters.sort(key=lambda x: len(x.points), reverse=True)
        largest_cluster = valid_clusters[0]
        
        obb = largest_cluster.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        geometries_to_draw.extend(valid_clusters)
        geometries_to_draw.append(obb)

        center = obb.get_center()
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "camera_rgb_optical_frame"
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = center
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = obb.R
        quaternion = quaternion_from_matrix(transform_matrix)
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quaternion

        status = f"Found {len(valid_clusters)} object(s). Largest at Z={center[2]:.2f}m"
        return pose_msg, status, geometries_to_draw


class ObjectDetectorGUI:
    def __init__(self):
        rospy.init_node("object_detector_gui_node", anonymous=True)
        self.bridge = CvBridge()
        self.is_shutting_down = False
        self._data_lock = threading.Lock()

        self.root = tk.Tk()
        self.root.title("Interactive Object Detector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._setup_gui()

        self._latest_rgb = None
        self._latest_depth = None
        self.detector_logic = None
        self.pose_publisher = rospy.Publisher(OBJECT_POSE_TOPIC, PoseStamped, queue_size=1)
        self._initialize_ros_and_logic()
        
        self._gui_update_loop()
        rospy.loginfo("Object Detector GUI Initialized.")

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(main_frame, text="Waiting for camera feed...")
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.detect_button = ttk.Button(main_frame, text="Detect Objects and Show 3D", command=self._trigger_detection_thread)
        self.detect_button.pack(fill=tk.X, ipady=10, pady=5)
        
        self.status_var = tk.StringVar(value="Status: Ready. Press button to detect.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _initialize_ros_and_logic(self):
        try:
            info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=5)
            self.detector_logic = ObjectDetectorLogic(camera_intrinsics=info_msg)
            
            rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
            depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
            ts.registerCallback(self.ros_callback)
        except Exception as e:
            messagebox.showerror("ROS Error", f"Could not connect to camera topics: {e}")
            self.on_close()

    def ros_callback(self, rgb_msg, depth_msg):
        with self._data_lock:
            self._latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self._latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

    def _trigger_detection_thread(self):
        self.detect_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Processing...")
        threading.Thread(target=self._run_detection_pipeline, daemon=True).start()

    def _run_detection_pipeline(self):
        with self._data_lock:
            if self._latest_depth is None or self._latest_rgb is None:
                self.status_var.set("Status: Error - No image data available.")
                self.detect_button.config(state=tk.NORMAL)
                return
            depth_image, rgb_image = self._latest_depth.copy(), self._latest_rgb.copy()

        try:
            pcd = self.detector_logic.create_pcd_from_rgbd_manual(rgb_image, depth_image)
            pose_msg, status_text, geoms_to_draw = self.detector_logic.find_objects(pcd)
            
            if pose_msg:
                pose_msg.header.stamp = rospy.Time.now()
                self.pose_publisher.publish(pose_msg)
                rospy.loginfo(f"Published object pose: {status_text}")
            
            self.status_var.set(f"Status: {status_text}. Close 3D window to continue.")
            
            if geoms_to_draw:
                o3d.visualization.draw_geometries(geoms_to_draw, window_name="3D Detection Result")

        except Exception as e:
            rospy.logerr(f"Object detection pipeline failed: {e}")
            self.status_var.set(f"Status: Error - {e}")
        finally:
            self.status_var.set("Status: Ready.")
            self.detect_button.config(state=tk.NORMAL)

    def _gui_update_loop(self):
        with self._data_lock:
            if self._latest_rgb is not None:
                self.update_video_display(self._latest_rgb)
        
        if not self.is_shutting_down:
            self.root.after(40, self._gui_update_loop)

    def update_video_display(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

    def on_close(self):
        self.is_shutting_down = True
        rospy.signal_shutdown("GUI Closed")
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    try:
        app = ObjectDetectorGUI()
        app.run()
    except Exception as e:
        rospy.logfatal(f"Unhandled exception: {e}")