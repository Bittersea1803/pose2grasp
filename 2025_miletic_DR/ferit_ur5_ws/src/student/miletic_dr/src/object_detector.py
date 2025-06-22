#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Vector3
from cv_bridge import CvBridge
import message_filters
from tf.transformations import quaternion_from_matrix

from miletic_dr.msg import DetectedObject

#TODO: Čita 

# CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
# CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
# CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"


OBJECT_DATA_TOPIC = "/miletic_dr/detected_object" 
FRAME_ID = "camera_rgb_optical_frame"

VOXEL_SIZE = 0.005
PLANE_DIST_THRESHOLD = 0.015
DBSCAN_EPS = 0.025
DBSCAN_MIN_POINTS = 50

class ObjectDetectorLogic:
    def __init__(self, camera_intrinsics):
        self.fx = camera_intrinsics.K[0]
        self.fy = camera_intrinsics.K[4]
        self.cx = camera_intrinsics.K[2]
        self.cy = camera_intrinsics.K[5]

    def create_pcd_from_rgbd(self, rgb_image, depth_image):
        depth_16u = np.asarray(depth_image)
        rgb_bgr = np.asarray(rgb_image)
        valid_mask = (depth_16u > 400) & (depth_16u < 2000)
        indices = np.array(np.nonzero(valid_mask)).T
        if indices.shape[0] == 0: return None
        
        us = indices[:, 1].astype(np.float32); vs = indices[:, 0].astype(np.float32)
        zs = depth_16u[indices[:, 0], indices[:, 1]].astype(np.float32) / 1000.0
        xs = (us - self.cx) * zs / self.fx; ys = (vs - self.cy) * zs / self.fy
        
        points = np.stack((xs, ys, zs), axis=-1)
        colors_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[indices[:, 0], indices[:, 1]]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
        return pcd

    def find_largest_object_data(self, pcd):
        if not pcd or not pcd.has_points(): return None

        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        
        try:
            plane_model, inliers = pcd_down.segment_plane(PLANE_DIST_THRESHOLD, 3, 1000)
            plane_cloud = pcd_down.select_by_index(inliers)
            objects_cloud = pcd_down.select_by_index(inliers, invert=True)
            plane_detected = True
        except RuntimeError:
            objects_cloud = pcd_down
            plane_detected = False

        if not objects_cloud.has_points(): return None

        labels = np.array(objects_cloud.cluster_dbscan(DBSCAN_EPS, DBSCAN_MIN_POINTS, print_progress=False))
        if labels.max() < 0: return None

        counts = np.bincount(labels[labels >= 0])
        largest_cluster_label = np.argmax(counts)
        largest_cluster = objects_cloud.select_by_index(np.where(labels == largest_cluster_label)[0])

        if plane_detected:
            plane_normal = np.array(plane_model[:3])
            if np.dot(plane_normal, -plane_cloud.get_center()) < 0: plane_normal = -plane_normal
            z_axis = plane_normal / np.linalg.norm(plane_normal)
            
            ref_vec = np.array([1., 0., 0.]);
            if np.abs(np.dot(z_axis, ref_vec)) > 0.95: ref_vec = np.array([0., 1., 0.])
            y_axis_temp = np.cross(z_axis, ref_vec); y_axis_temp /= np.linalg.norm(y_axis_temp)
            x_axis_temp = np.cross(y_axis_temp, z_axis); x_axis_temp /= np.linalg.norm(x_axis_temp)
            R_world_to_plane_temp = np.stack([x_axis_temp, y_axis_temp, z_axis], axis=0)
            
            transformed_points = (R_world_to_plane_temp @ np.asarray(largest_cluster.points).T).T
            points_2d = transformed_points[:, :2].astype(np.float32)
            rect_2d = cv2.minAreaRect(points_2d)
            center_2d, (width, height), angle_deg = rect_2d

            if width < height:
                angle_for_y_axis_deg = angle_deg; obb_extent_2d = [width, height]
            else:
                angle_for_y_axis_deg = angle_deg + 90; obb_extent_2d = [height, width]

            angle_rad = np.deg2rad(angle_for_y_axis_deg)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R_on_plane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            R_plane_to_world = R_world_to_plane_temp.T
            final_rotation = R_plane_to_world @ R_on_plane
            
            plane_level_z = np.mean((R_world_to_plane_temp @ np.asarray(plane_cloud.points).T).T[:, 2])
            max_z = np.max(transformed_points[:, 2])
            extent = np.array([obb_extent_2d[0], obb_extent_2d[1], max_z - plane_level_z])
            center_in_plane_coords = np.array([center_2d[0], center_2d[1], plane_level_z + extent[2] / 2])
            center_world = R_plane_to_world @ center_in_plane_coords
            
            return center_world, final_rotation, extent
        else:
            obb = largest_cluster.get_oriented_bounding_box()
            return obb.get_center(), obb.R, obb.extent


class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node('object_detector_node')
        
        try:
            cam_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
        except rospy.ROSException:
            rospy.logfatal("Timeout waiting for camera_info. Is the camera running?")
            rospy.signal_shutdown("Camera info not available.")
            return

        self.logic = ObjectDetectorLogic(camera_intrinsics=cam_info)
        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher(OBJECT_DATA_TOPIC, DetectedObject, queue_size=1)

        rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        rospy.loginfo("Object Detector node initialized and running.")

    def image_callback(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Failed to convert images: {e}")
            return

        pcd = self.logic.create_pcd_from_rgbd(rgb, depth)
        result = self.logic.find_largest_object_data(pcd)

        if result:
            center, rotation_matrix, dimensions = result
            
            msg = DetectedObject()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = FRAME_ID
            
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = center[0], center[1], center[2]
            
            transform_for_quat = np.identity(4)
            transform_for_quat[:3,:3] = rotation_matrix
            q = quaternion_from_matrix(transform_for_quat)
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q[0], q[1], q[2], q[3]

            msg.dimensions.x, msg.dimensions.y, msg.dimensions.z = dimensions[0], dimensions[1], dimensions[2]

            self.pose_pub.publish(msg)
            rospy.loginfo_once("Published object pose and dimensions.")


if __name__ == '__main__':
    try:
        ObjectDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
