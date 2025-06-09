#!/usr/bin/env python
import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped

# --- Hardcoded Configuration ---
INPUT_TOPIC = "/camera/depth_registered/points"
OUTPUT_TOPIC = "/object_detector/object_position"
VOXEL_LEAF_SIZE = 0.005
PLANE_DISTANCE_THRESHOLD = 0.015
CLUSTER_EPS = 0.03
CLUSTER_MIN_POINTS = 40

def convert_ros_to_o3d(ros_cloud):
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names))
    if len(cloud_data) == 0: return None
    xyz = [(p[0], p[1], p[2]) for p in cloud_data]
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    return o3d_cloud

class ObjectDetectorOpen3D:
    def __init__(self):
        rospy.loginfo("Initializing Plane Object Detector (Standalone)...")
        self.publisher = rospy.Publisher(OUTPUT_TOPIC, PointStamped, queue_size=10)
        rospy.Subscriber(INPUT_TOPIC, PointCloud2, self.cloud_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("Plane Object Detector (Open3D) is running.")

    def cloud_callback(self, ros_cloud):
        o3d_cloud = convert_ros_to_o3d(ros_cloud)
        if o3d_cloud is None: return

        o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=VOXEL_LEAF_SIZE)

        try:
            plane_model, inliers = o3d_cloud.segment_plane(distance_threshold=PLANE_DISTANCE_THRESHOLD,
                                                           ransac_n=3,
                                                           num_iterations=1000)
        except Exception: return
        
        object_cloud = o3d_cloud.select_by_index(inliers, invert=True)
        if not object_cloud.has_points(): return

        labels = np.array(object_cloud.cluster_dbscan(eps=CLUSTER_EPS, 
                                                       min_points=CLUSTER_MIN_POINTS, 
                                                       print_progress=False))
        
        if labels.max() < 0: return

        counts = np.bincount(labels[labels >= 0])
        largest_cluster_label = counts.argmax()
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        largest_cluster_cloud = object_cloud.select_by_index(largest_cluster_indices)
        centroid = largest_cluster_cloud.get_center()

        point_msg = PointStamped()
        point_msg.header = ros_cloud.header
        point_msg.point.x, point_msg.point.y, point_msg.point.z = centroid
        self.publisher.publish(point_msg)

if __name__ == '__main__':
    rospy.init_node('plane_object_detector_open3d')
    try:
        ObjectDetectorOpen3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
