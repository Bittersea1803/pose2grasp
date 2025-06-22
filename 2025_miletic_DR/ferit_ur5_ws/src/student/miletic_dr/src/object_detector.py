#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import open3d as o3d
import json
import os
import sys

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from core.real_ur5_controller import UR5Controller

CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"
OUTPUT_FILENAME = "object_data.json"

SCAN_POSE_JOINTS = np.deg2rad([-91.13, -98.58, -22.66, -143.08, 92.83, 45.45])

VOXEL_SIZE = 0.005              
PLANE_DIST_THRESHOLD = 0.015    
DBSCAN_EPS = 0.025              
DBSCAN_MIN_POINTS = 50          
DEPTH_RANGE_MIN = 0.4           
DEPTH_RANGE_MAX = 2.0           

class ObjectDetectorLogic:
    def __init__(self, camera_intrinsics):
        self.fx = camera_intrinsics.K[0]
        self.fy = camera_intrinsics.K[4]
        self.cx = camera_intrinsics.K[2]
        self.cy = camera_intrinsics.K[5]
        rospy.loginfo("ObjectDetectorLogic initialized with camera intrinsics.")

    def create_pcd_from_rgbd(self, rgb_image, depth_image):
        depth_16u = np.asarray(depth_image)
        rgb_bgr = np.asarray(rgb_image)
        
        valid_mask = (depth_16u > (DEPTH_RANGE_MIN * 1000)) & (depth_16u < (DEPTH_RANGE_MAX * 1000))
        indices = np.array(np.nonzero(valid_mask)).T
        if indices.shape[0] == 0:
            rospy.logwarn("No valid depth points found in the specified range.")
            return None
        
        us = indices[:, 1].astype(np.float32)
        vs = indices[:, 0].astype(np.float32)
        zs = depth_16u[indices[:, 0], indices[:, 1]].astype(np.float32) / 1000.0
        
        xs = (us - self.cx) * zs / self.fx
        ys = (vs - self.cy) * zs / self.fy
        
        points = np.stack((xs, ys, zs), axis=-1)
        colors_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[indices[:, 0], indices[:, 1]]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
        rospy.loginfo(f"Point cloud created with {len(pcd.points)} points.")
        return pcd

    def find_largest_object_data(self, pcd):
        if not pcd or not pcd.has_points():
            rospy.logwarn("Cannot find object, input point cloud is empty.")
            return None

        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        rospy.loginfo(f"Point cloud downsampled to {len(pcd_down.points)} points.")
        
        #   1. Plane Segmentation  
        try:
            plane_model, inliers = pcd_down.segment_plane(PLANE_DIST_THRESHOLD, 3, 1000)
            plane_cloud = pcd_down.select_by_index(inliers)
            objects_cloud = pcd_down.select_by_index(inliers, invert=True)
            plane_detected = True
            rospy.loginfo(f"Plane detected with {len(inliers)} inlier points. {len(objects_cloud.points)} object points remaining.")
        except RuntimeError:
            objects_cloud = pcd_down
            plane_detected = False
            rospy.logwarn("Plane segmentation failed, treating entire cloud as objects.")

        if not objects_cloud.has_points():
            rospy.logwarn("No object points remaining after plane removal.")
            return None

        #   2. Clustering  
        labels = np.array(objects_cloud.cluster_dbscan(DBSCAN_EPS, DBSCAN_MIN_POINTS, print_progress=False))
        if labels.max() < 0:
            rospy.logwarn("DBSCAN found no valid clusters meeting the size requirement.")
            return None

        counts = np.bincount(labels[labels >= 0])
        largest_cluster_label = np.argmax(counts)
        largest_cluster = objects_cloud.select_by_index(np.where(labels == largest_cluster_label)[0])
        rospy.loginfo(f"Found {labels.max() + 1} clusters. Largest has {len(largest_cluster.points)} points.")

        #   3. Bounding Box Calculation  
        obb = largest_cluster.get_oriented_bounding_box()
        
        #   4. Axis Alignment for Consistency  
        # Ensure the Z-axis of the box points "up" (away from the table).
        if plane_detected:
            plane_normal = np.array(plane_model[:3])
            if np.dot(plane_normal, -plane_cloud.get_center()) < 0:
                plane_normal = -plane_normal
            
            if np.dot(obb.R[:, 2], plane_normal) < 0:
                rospy.loginfo("Aligning Axes: OBB Z-axis was pointing down, flipping it.")
                obb.R[:, 2] = -obb.R[:, 2]
                obb.R[:, 1] = -obb.R[:, 1]
        
        # Ensure the Y-axis corresponds to the longest side of the object base.
        if obb.extent[0] > obb.extent[1]:
            rospy.loginfo("Aligning Axes: Swapping X and Y to make Y the longest dimension.")
            x_axis, y_axis = obb.R[:, 0].copy(), obb.R[:, 1].copy()
            obb.R[:, 0], obb.R[:, 1] = y_axis, x_axis
            obb.extent[0], obb.extent[1] = obb.extent[1], obb.extent[0]

        center = obb.get_center()
        rotation_matrix = obb.R
        dimensions = obb.extent
        
        rospy.loginfo("  DETECTED OBJECT PROPERTIES  ")
        rospy.loginfo(f"Center (m):          {np.round(center, 4)}")
        rospy.loginfo(f"Dimensions (X,Y,Z m): {np.round(dimensions, 4)}")
        rospy.loginfo(f"Rotation Matrix:\n{np.round(rotation_matrix, 3)}")
        
        return center, rotation_matrix, dimensions

class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node('object_detector_node', anonymous=True)
        rospy.loginfo("Initializing Object Detector Node...")
        
        self.logic = None
        self.bridge = CvBridge()
        self.robot_controller = UR5Controller()
        self.task_complete = False

    def run(self):
        # 1. Move robot to SCAN position
        rospy.loginfo("Step 1: Moving robot to SCAN_POSE for object detection.")
        joint_trajectory_points = np.array([
            self.robot_controller.get_current_joint_values(), 
            SCAN_POSE_JOINTS
        ])
        self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)
        rospy.loginfo("Robot is at SCAN_POSE.")

        # 2. Get camera info first
        try:
            rospy.loginfo("Step 2: Waiting for camera info...")
            cam_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
            self.logic = ObjectDetectorLogic(camera_intrinsics=cam_info)
        except rospy.ROSException:
            rospy.logfatal("Timeout waiting for camera_info. Is the camera running?")
            return

        # 3. Get a single pair of synced images
        try:
            rospy.loginfo("Step 3: Waiting for a synchronized RGB and Depth image pair...")
            rgb_msg = rospy.wait_for_message(CAMERA_RGB_TOPIC, Image, timeout=10)
            depth_msg = rospy.wait_for_message(CAMERA_DEPTH_TOPIC, Image, timeout=10)
        except rospy.ROSException as e:
            rospy.logfatal(f"Timeout waiting for images: {e}")
            return
            
        rospy.loginfo("Image pair received. Starting detection...")

        # 4. Process the images
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Failed to convert images: {e}")
            return
            
        pcd = self.logic.create_pcd_from_rgbd(rgb_image, depth_image)
        result = self.logic.find_largest_object_data(pcd)

        # 5. Save results and shutdown
        if result:
            center, rotation_matrix, dimensions = result
            self.write_result_and_shutdown(center, rotation_matrix, dimensions)
        else:
            rospy.logerr("Object detection failed. No result to save. Shutting down.")
            rospy.signal_shutdown("Object detection failed.")
    
    def write_result_and_shutdown(self, center, rotation_matrix, dimensions):
        if self.task_complete: return
        self.task_complete = True
        
        rospy.loginfo(f"  OBJECT DETECTED  ")
        
        output_data = {
            "center": center.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "dimensions": dimensions.tolist()
        }

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, OUTPUT_FILENAME)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            rospy.loginfo(f"Successfully wrote object data to '{output_path}'.")
        except IOError as e:
            rospy.logfatal(f"Failed to write to file {output_path}: {e}")
        
        rospy.loginfo("Shutting down node.")
        rospy.signal_shutdown("Object detected and saved to file.")

if __name__ == '__main__':
    try:
        node = ObjectDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ObjectDetectorNode shutdown complete.")
