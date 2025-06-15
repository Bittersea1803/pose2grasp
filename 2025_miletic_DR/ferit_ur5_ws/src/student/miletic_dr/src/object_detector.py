#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
from tf.transformations import quaternion_from_matrix

# --- Camera Topic Configuration ---
CAMERA_RGB_TOPIC = "/camera/rgb/image_rect_color"
CAMERA_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

# --- Constants ---
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

    def find_largest_object(self, pcd):
        if not pcd.has_points():
            return None

        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        if not pcd_down.has_points():
            return None

        try:
            _, inliers = pcd_down.segment_plane(
                distance_threshold=PLANE_DIST_THRESHOLD, ransac_n=3, num_iterations=1000)
        except RuntimeError:
            return None

        objects_cloud = pcd_down.select_by_index(inliers, invert=True)
        if not objects_cloud.has_points():
            return None

        labels = np.array(objects_cloud.cluster_dbscan(
            eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False))

        max_label = labels.max()
        if max_label < 0:
            return None

        counts = np.bincount(labels[labels >= 0])
        largest_cluster_label = np.argmax(counts)
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        largest_cluster = objects_cloud.select_by_index(largest_cluster_indices)

        if len(largest_cluster.points) < DBSCAN_MIN_POINTS:
            return None

        obb = largest_cluster.get_oriented_bounding_box()

        center = obb.get_center()
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = FRAME_ID
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = center

        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = obb.R
        quat = quaternion_from_matrix(transform_matrix)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        return pose_msg


class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node('object_detector_node')

        try:
            cam_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
        except rospy.ROSException:
            rospy.logfatal("Timeout waiting for camera_info. Is the camera publishing?")
            rospy.signal_shutdown("Camera info not available.")
            return

        self.logic = ObjectDetectorLogic(camera_intrinsics=cam_info)
        self.bridge = CvBridge()

        self.pose_pub = rospy.Publisher("/plane_object_detector/biggest_object_position", PoseStamped, queue_size=1)

        rgb_sub = message_filters.Subscriber(CAMERA_RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(CAMERA_DEPTH_TOPIC, Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        rospy.loginfo("Object Detector Node initialized and running.")

    def image_callback(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Failed to convert images: {e}")
            return

        pcd = self.logic.create_pcd_from_rgbd_manual(rgb, depth)
        pose_msg = self.logic.find_largest_object(pcd)

        if pose_msg:
            pose_msg.header.stamp = rospy.Time.now()
            self.pose_pub.publish(pose_msg)
            rospy.loginfo("Published object pose.")


if __name__ == '__main__':
    try:
        ObjectDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
