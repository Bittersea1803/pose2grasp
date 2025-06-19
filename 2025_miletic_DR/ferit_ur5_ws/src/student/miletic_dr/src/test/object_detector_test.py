#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import numpy as np
import cv2  # Provjerite je li cv2 importan
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

# Definicije ROS topica
CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"
OBJECT_POSE_TOPIC = "/plane_object_detector/biggest_object_position"


class ObjectDetectorLogic:
    def __init__(self, camera_intrinsics):
        self.fx = camera_intrinsics.K[0]
        self.fy = camera_intrinsics.K[4]
        self.cx = camera_intrinsics.K[2]
        self.cy = camera_intrinsics.K[5]
        
        # Parametri za detekciju
        self.VOXEL_SIZE = 0.005
        self.PLANE_DIST_THRESHOLD = 0.015
        self.DBSCAN_EPS = 0.025
        self.DBSCAN_MIN_POINTS = 50

    def create_pcd_from_rgbd_manual(self, rgb_image, depth_image):
        depth_16u = np.asarray(depth_image)
        rgb_bgr = np.asarray(rgb_image)
        H, W = depth_16u.shape
        # Filtriranje dubinskih podataka po udaljenosti
        valid_mask = (depth_16u > 400) & (depth_16u < 2000)
        indices = np.array(np.nonzero(valid_mask)).T
        if indices.shape[0] == 0:
            return o3d.geometry.PointCloud()

        # Kreiranje 3D točaka iz dubinske slike (pinhole model kamere)
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
            return None, "Point cloud je prazan", []

        pcd_down = pcd.voxel_down_sample(self.VOXEL_SIZE)
        if not pcd_down.has_points():
            return None, "Point cloud nakon downsamplinga je prazan", []

        # Detekcija ravnine (stola) pomoću RANSAC algoritma
        plane_detected = False
        plane_cloud = o3d.geometry.PointCloud()
        try:
            plane_model, inliers = pcd_down.segment_plane(
                distance_threshold=self.PLANE_DIST_THRESHOLD, ransac_n=3, num_iterations=1000)
            
            if len(inliers) > self.DBSCAN_MIN_POINTS:
                plane_detected = True
                plane_cloud = pcd_down.select_by_index(inliers)
                objects_cloud = pcd_down.select_by_index(inliers, invert=True)
                plane_cloud.paint_uniform_color([0.7, 0.7, 0.7]) # Siva boja za ravninu
                geometries_to_draw = [plane_cloud]
            else:
                objects_cloud = pcd_down
                geometries_to_draw = []
        except RuntimeError:
            objects_cloud = pcd_down
            geometries_to_draw = []

        if not objects_cloud.has_points():
            return None, "Nema točaka nakon uklanjanja ravnine", geometries_to_draw

        # Klasteriranje preostalih točaka da se pronađu objekti
        labels = np.array(objects_cloud.cluster_dbscan(
            eps=self.DBSCAN_EPS, min_points=self.DBSCAN_MIN_POINTS, print_progress=False))
        
        max_label = labels.max()
        if max_label < 0:
            return None, "DBSCAN nije pronašao klastere", geometries_to_draw

        # Izdvajanje validnih klastera (objekata)
        valid_clusters = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > self.DBSCAN_MIN_POINTS:
                cluster_cloud = objects_cloud.select_by_index(cluster_indices)
                cluster_cloud.paint_uniform_color(np.random.rand(3))
                valid_clusters.append(cluster_cloud)
        
        if not valid_clusters:
            return None, "Nema validnih klastera", geometries_to_draw

        # Pronalazak najvećeg klastera
        valid_clusters.sort(key=lambda x: len(x.points), reverse=True)
        largest_cluster = valid_clusters[0]
        
        ### START: ISPRAVLJENI BLOK ZA KREIRANJE BOUNDING BOX-a ###
        if plane_detected:
            # KORAK 1: Definiraj koordinatni sustav poravnat s ravninom
            plane_normal = np.array(plane_model[:3])
            # Osiguraj da normala gleda "prema gore" (prema kameri)
            if np.dot(plane_normal, -plane_cloud.get_center()) < 0:
                plane_normal = -plane_normal
            
            z_axis = plane_normal / np.linalg.norm(plane_normal)
            
            # Odaberi referentni vektor za kreiranje ortonormirane baze
            ref_vec = np.array([1., 0., 0.])
            if np.abs(np.dot(z_axis, ref_vec)) > 0.95:
                ref_vec = np.array([0., 1., 0.])
            
            # Kreiraj X i Y osi koje su okomite na Z i jedna na drugu (desni koordinatni sustav)
            y_axis = np.cross(z_axis, ref_vec)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            
            # Matrica rotacije koja transformira iz SVIJETA u sustav RAVNINE
            # p_plane = R_world_to_plane @ p_world
            R_world_to_plane = np.stack([x_axis, y_axis, z_axis], axis=0)
            # Inverzna matrica, transformira iz RAVNINE u SVIJET
            # p_world = R_plane_to_world @ p_plane
            R_plane_to_world = R_world_to_plane.T

            # KORAK 2: Transformiraj točke objekta u novi koordinatni sustav
            cluster_points = np.asarray(largest_cluster.points)
            # Koristimo standardnu (pre-multiply) konvenciju: p' = R @ p
            # Stoga točke moraju biti stupci (3, N), pa ih transponiramo
            transformed_points = (R_world_to_plane @ cluster_points.T).T

            # KORAK 3: Nađi najuži 2D okvir (bounding box) na XY-ravnini novog sustava
            points_2d = transformed_points[:, :2].astype(np.float32)
            if len(points_2d) < 3:
                final_obb = largest_cluster.get_oriented_bounding_box()
            else:
                rect_2d = cv2.minAreaRect(points_2d)
                center_2d, size_2d, angle_deg = rect_2d

                # KORAK 4: Kreiraj 3D rotaciju i poziciju iz 2D podataka
                # Kut rotacije na ravnini (oko Z osi)
                angle_rad = np.deg2rad(angle_deg)
                R_on_plane = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad),  np.cos(angle_rad), 0],
                    [0, 0, 1]
                ])

                # Finalna rotacija je kombinacija rotacije ravnine i rotacije na ravnini
                # p_world = R_plane_to_world @ R_on_plane @ p_local
                final_rotation = R_plane_to_world @ R_on_plane

                # KORAK 5: Izračunaj dimenzije (extent) i centar 3D okvira
                # Visina se računa od ravnine do vrha objekta
                plane_center_transformed = R_world_to_plane @ plane_cloud.get_center()
                plane_level_z = plane_center_transformed[2]
                
                # Vrh objekta u sustavu ravnine
                max_z = np.max(transformed_points[:, 2])
                
                # Dimenzije. cv2 može zamijeniti width/height, pa koristimo size_2d
                width, height = size_2d
                extent = np.array([width, height, max_z - plane_level_z])
                
                # Centar okvira u koordinatama RAVNINE
                center_in_plane_coords = np.array([
                    center_2d[0],
                    center_2d[1],
                    plane_level_z + extent[2] / 2
                ])
                
                # Transformiraj centar okvira natrag u SVJETSKE koordinate
                center_world = R_plane_to_world @ center_in_plane_coords

                final_obb = o3d.geometry.OrientedBoundingBox(center_world, final_rotation, extent)
        else:
            # Fallback ako ravnina nije pronađena
            final_obb = largest_cluster.get_oriented_bounding_box()
        ### END: ISPRAVLJENI BLOK ###
        
        final_obb.color = (0, 1, 0) # Zelena boja za BBox
        geometries_to_draw.extend(valid_clusters)
        geometries_to_draw.append(final_obb)
        
        # Iscrtavanje centra i koordinatnog sustava BBox-a
        center = final_obb.get_center()
        center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        center_marker.translate(center)
        center_marker.paint_uniform_color([1, 0, 0])
        geometries_to_draw.append(center_marker)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        coord_frame.rotate(final_obb.R, center=[0, 0, 0])
        coord_frame.translate(center)
        geometries_to_draw.append(coord_frame)

        # Kreiranje PoseStamped poruke za ROS
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "camera_rgb_optical_frame"
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = center
        
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = final_obb.R
        quaternion = quaternion_from_matrix(transform_matrix)
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quaternion

        status = f"Pronađeno {len(valid_clusters)} objekata. Najveći na Z={center[2]:.2f}m"
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
        self.pose_publisher = rospy.Publisher(OBJECT_POSE_TOPIC, PoseStamped, queue_size=10)
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
            rospy.loginfo("Waiting for camera info...")
            info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
            self.detector_logic = ObjectDetectorLogic(camera_intrinsics=info_msg)
            
            rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
            depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
            ts.registerCallback(self.ros_callback)
            rospy.loginfo("Camera topics connected.")
        except Exception as e:
            rospy.logerr(f"Could not connect to camera topics: {e}")
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
                self.root.after(0, lambda: self.status_var.set("Status: Error - No image data available."))
                self.root.after(0, lambda: self.detect_button.config(state=tk.NORMAL))
                return
            depth_image, rgb_image = self._latest_depth.copy(), self._latest_rgb.copy()

        try:
            pcd = self.detector_logic.create_pcd_from_rgbd_manual(rgb_image, depth_image)
            pose_msg, status_text, geoms_to_draw = self.detector_logic.find_objects(pcd)
            
            if pose_msg:
                pose_msg.header.stamp = rospy.Time.now()
                self.pose_publisher.publish(pose_msg)
                rospy.loginfo(f"Published object pose: {status_text}")
            
            self.root.after(0, lambda: self.status_var.set(f"Status: {status_text}. Close 3D window to continue."))
            
            if geoms_to_draw:
                o3d.visualization.draw_geometries(geoms_to_draw, window_name="3D Detection Result")

        except Exception as e:
            rospy.logerr(f"Object detection pipeline failed: {e}", exc_info=True)
            self.root.after(0, lambda: self.status_var.set(f"Status: Error - {e}"))
        finally:
            self.root.after(0, lambda: self.status_var.set("Status: Ready."))
            self.root.after(0, lambda: self.detect_button.config(state=tk.NORMAL))

    def _gui_update_loop(self):
        if self.is_shutting_down:
            return
        with self._data_lock:
            if self._latest_rgb is not None:
                self.update_video_display(self._latest_rgb)
        
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
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    try:
        app = ObjectDetectorGUI()
        app.run()
    except Exception as e:
        rospy.logfatal(f"Unhandled exception: {e}")