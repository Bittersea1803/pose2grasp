#!/usr/bin/env python3More actions
# -*- coding: utf-8 -*-

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

GRIPPER_MESH_PATH = "mesh.ply"

# --- ROS Topici - HOME ---
CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"
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

        try:
            self.gripper_mesh = o3d.io.read_triangle_mesh(GRIPPER_MESH_PATH)
            # self.gripper_mesh.translate(-self.gripper_mesh.get_center(), relative=True) VERY BAD IDEA
            self.gripper_mesh_loaded = True
            rospy.loginfo(f"3D model hvataljke ({GRIPPER_MESH_PATH}) uspješno učitan.")
        except Exception as e:
            self.gripper_mesh_loaded = False
            rospy.logwarn(f"Nije moguće učitati 3D model hvataljke ({GRIPPER_MESH_PATH}): {e}")
            self.gripper_mesh = None

    def create_pcd_from_rgbd_manual(self, rgb_image, depth_image):
        depth_16u = np.asarray(depth_image)
        rgb_bgr = np.asarray(rgb_image)
        valid_mask = (depth_16u > 400) & (depth_16u < 2000)
        indices = np.array(np.nonzero(valid_mask)).T
        if indices.shape[0] == 0: return o3d.geometry.PointCloud()
        us = indices[:, 1].astype(np.float32); vs = indices[:, 0].astype(np.float32)
        zs = depth_16u[indices[:, 0], indices[:, 1]].astype(np.float32) / 1000.0
        xs = (us - self.cx) * zs / self.fx; ys = (vs - self.cy) * zs / self.fy
        points = np.stack((xs, ys, zs), axis=-1)
        colors_bgr = rgb_bgr[indices[:, 0], indices[:, 1]]
        colors_rgb = colors_bgr[:, ::-1] / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        return pcd

    def find_objects(self, pcd):
        if not pcd.has_points(): return None, "Point cloud je prazan", []

        pcd_down = pcd.voxel_down_sample(self.VOXEL_SIZE)
        geometries_to_draw = []

        # Detekcija ravnine
        plane_detected = False
        try:
            plane_model, inliers = pcd_down.segment_plane(self.PLANE_DIST_THRESHOLD, 3, 1000)
            if len(inliers) > self.DBSCAN_MIN_POINTS:
                plane_detected = True
                plane_cloud = pcd_down.select_by_index(inliers)
                objects_cloud = pcd_down.select_by_index(inliers, invert=True)
                plane_cloud.paint_uniform_color([0.7, 0.7, 0.7])
                geometries_to_draw.append(plane_cloud)
            else: objects_cloud = pcd_down
        except RuntimeError: objects_cloud = pcd_down

        if not objects_cloud.has_points(): return None, "Nema točaka nakon uklanjanja ravnine", geometries_to_draw

        # Klasteriranje
        labels = np.array(objects_cloud.cluster_dbscan(self.DBSCAN_EPS, self.DBSCAN_MIN_POINTS))
        max_label = labels.max()
        if max_label < 0: return None, "DBSCAN nije pronašao klastere", geometries_to_draw

        valid_clusters = [objects_cloud.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)]
        valid_clusters.sort(key=lambda x: len(x.points), reverse=True)
        largest_cluster = valid_clusters[0]

        # --- IZRACUN BOUNDING BOX
        if plane_detected:
            # Z-os je paralelna s normalom ravnine
            plane_normal = np.array(plane_model[:3])
            if np.dot(plane_normal, -plane_cloud.get_center()) < 0: plane_normal = -plane_normal
            z_axis = plane_normal / np.linalg.norm(plane_normal)

            # Pomoćna rotacija za transformaciju u 2D
            ref_vec = np.array([1., 0., 0.]);
            if np.abs(np.dot(z_axis, ref_vec)) > 0.95: ref_vec = np.array([0., 1., 0.])
            y_axis_temp = np.cross(z_axis, ref_vec); y_axis_temp /= np.linalg.norm(y_axis_temp)
            x_axis_temp = np.cross(y_axis_temp, z_axis); x_axis_temp /= np.linalg.norm(x_axis_temp)
            R_world_to_plane_temp = np.stack([x_axis_temp, y_axis_temp, z_axis], axis=0)

            transformed_points = (R_world_to_plane_temp @ np.asarray(largest_cluster.points).T).T
            points_2d = transformed_points[:, :2].astype(np.float32)
            rect_2d = cv2.minAreaRect(points_2d)
            center_2d, (width, height), angle_deg = rect_2d

            # Y-os objekta je poravnata s dužom stranicom
            if width < height:
                angle_for_y_axis_deg = angle_deg
                obb_extent_2d = [width, height]
            else:
                angle_for_y_axis_deg = angle_deg + 90
                obb_extent_2d = [height, width]

            angle_rad = np.deg2rad(angle_for_y_axis_deg)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R_on_plane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            # Finalna rotacija i pozicija objekta
            R_plane_to_world = R_world_to_plane_temp.T
            final_rotation = R_plane_to_world @ R_on_plane

            plane_level_z = np.mean((R_world_to_plane_temp @ np.asarray(plane_cloud.points).T).T[:, 2])
            max_z = np.max(transformed_points[:, 2])
            extent = np.array([obb_extent_2d[0], obb_extent_2d[1], max_z - plane_level_z])
            center_in_plane_coords = np.array([center_2d[0], center_2d[1], plane_level_z + extent[2] / 2])
            center_world = R_plane_to_world @ center_in_plane_coords

            final_obb = o3d.geometry.OrientedBoundingBox(center_world, final_rotation, extent)
        else:
            final_obb = largest_cluster.get_oriented_bounding_box()
            z_axis = np.array([0, 0, 1])

        rospy.loginfo("\n" + "="*30 + "\n--- METRIKA DETEKCIJE ---")
        rospy.loginfo(f"Centar objekta: {np.round(final_obb.get_center(), 3)}")
        rospy.loginfo(f"Dimenzije (X, Y, Z): {np.round(final_obb.extent, 3)}")
        rospy.loginfo(f"Rotacijska matrica objekta:\n{np.round(final_obb.R, 2)}")

        largest_cluster.paint_uniform_color([0, 0.4, 0.8])
        geometries_to_draw.append(largest_cluster)
        final_obb.color = (0, 1, 0)
        geometries_to_draw.append(final_obb)

        # KS objekta
        object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        obb_transform = np.identity(4)
        obb_transform[:3, :3] = final_obb.R
        obb_transform[:3, 3] = final_obb.get_center()
        object_frame.transform(obb_transform)
        geometries_to_draw.append(object_frame)

        # KS ravnine
        if plane_detected:
            plane_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            plane_transform = np.identity(4)
            plane_transform[:3, :3] = R_plane_to_world
            plane_transform[:3, 3] = plane_cloud.get_center()
            plane_frame.transform(plane_transform)
            geometries_to_draw.append(plane_frame)

        # Pozicioniranje hvataljke
        if self.gripper_mesh_loaded:
            gripper_vis = o3d.geometry.TriangleMesh(self.gripper_mesh)

            # Orijentacija hvataljke
            gripper_z = -z_axis
            gripper_y = final_obb.R[:, 1]
            gripper_x = np.cross(gripper_y, gripper_z)
            gripper_rotation = np.stack([gripper_x, gripper_y, gripper_z], axis=1)

            # Pozicija hvataljke
            box_up_vector = final_obb.R[:, 2]
            gripper_position = final_obb.get_center() + box_up_vector * (final_obb.extent[2] / 2 + 0.03)

            # Primjena transformacije
            gripper_transform = np.identity(4)
            gripper_transform[:3, :3] = gripper_rotation
            gripper_transform[:3, 3] = gripper_position
            gripper_vis.transform(gripper_transform)
            gripper_vis.paint_uniform_color([0.9, 0.2, 0.2])
            geometries_to_draw.append(gripper_vis)

            # KS hvataljke
            gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            gripper_frame.transform(gripper_transform)
            geometries_to_draw.append(gripper_frame)

            rospy.loginfo("--- METRIKA HVATALJKE ---")
            rospy.loginfo(f"Pozicija hvataljke: {np.round(gripper_position, 3)}")
            rospy.loginfo(f"Rotacijska matrica hvataljke:\n{np.round(gripper_rotation, 2)}\n" + "="*30)

        # ROS msg
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now(); pose_msg.header.frame_id = "camera_rgb_optical_frame"
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = final_obb.get_center()
        transform_for_quat = np.identity(4)
        transform_for_quat[:3,:3] = final_obb.R
        quat = quaternion_from_matrix(transform_for_quat)
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat

        status = f"Pronađen objekt. BBox na Z={final_obb.get_center()[2]:.2f}m"
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
        self._latest_rgb = None; self._latest_depth = None
        self.detector_logic = None
        self.pose_publisher = rospy.Publisher(OBJECT_POSE_TOPIC, PoseStamped, queue_size=10)
        self._initialize_ros_and_logic()
        self._gui_update_loop()
        rospy.loginfo("Object Detector GUI Initialized.")
    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(main_frame, text="Waiting for camera feed..."); self.video_label.pack(fill=tk.BOTH, expand=True, pady=5)
        self.detect_button = ttk.Button(main_frame, text="Detect Objects and Show 3D", command=self._trigger_detection_thread); self.detect_button.pack(fill=tk.X, ipady=10, pady=5)
        self.status_var = tk.StringVar(value="Status: Ready. Press button to detect.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W); status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    def _initialize_ros_and_logic(self):
        try:
            info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
            self.detector_logic = ObjectDetectorLogic(camera_intrinsics=info_msg)
            rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
            depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
            ts.registerCallback(self.ros_callback)
        except Exception as e:
            messagebox.showerror("ROS Error", f"Could not connect to camera topics: {e}"); self.on_close()
    def ros_callback(self, rgb_msg, depth_msg):
        with self._data_lock:
            self._latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self._latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
    def _trigger_detection_thread(self):
        self.detect_button.config(state=tk.DISABLED); self.status_var.set("Status: Processing...")
        threading.Thread(target=self._run_detection_pipeline, daemon=True).start()
    def _run_detection_pipeline(self):
        with self._data_lock:
            if self._latest_depth is None or self._latest_rgb is None:
                self.root.after(0, lambda: self.status_var.set("Status: Error - No image data available."))
                self.root.after(0, lambda: self.detect_button.config(state=tk.NORMAL)); return
            depth_image, rgb_image = self._latest_depth.copy(), self._latest_rgb.copy()
        try:
            pcd = self.detector_logic.create_pcd_from_rgbd_manual(rgb_image, depth_image)
            pose_msg, status_text, geoms_to_draw = self.detector_logic.find_objects(pcd)
            if pose_msg: self.pose_publisher.publish(pose_msg)
            self.root.after(0, lambda: self.status_var.set(f"Status: {status_text}. Close 3D window to continue."))
            if geoms_to_draw: o3d.visualization.draw_geometries(geoms_to_draw, window_name="3D Detection Result")
        except Exception as e:
            rospy.logerr(f"Object detection pipeline failed: {e}", exc_info=True)
            error_message = f"Status: Error - {e}"; self.root.after(0, lambda: self.status_var.set(error_message))
        finally:
            self.root.after(0, lambda: self.status_var.set("Status: Ready.")); self.root.after(0, lambda: self.detect_button.config(state=tk.NORMAL))
    def _gui_update_loop(self):
        if self.is_shutting_down: return
        with self._data_lock:
            if self._latest_rgb is not None: self.update_video_display(self._latest_rgb)
        self.root.after(40, self._gui_update_loop)
    def update_video_display(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB); img_pil = PILImage.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil); self.video_label.configure(image=img_tk); self.video_label.image = img_tk
    def on_close(self):
        self.is_shutting_down = True; rospy.signal_shutdown("GUI Closed"); self.root.quit(); self.root.destroy()
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = ObjectDetectorGUI()
    app.run()