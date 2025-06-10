import rospy
import numpy as np
import open3d as o3d
from ctypes import c_uint32, c_float
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from geometry_msgs.msg import PointStamped

def convert_ros_to_o3d(ros_cloud):
    # Get cloud data from ros_cloud
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(rospy.numpy_msg.numpy_msg.point_cloud2.read_points(ros_cloud, skip_nans=True, field_names=field_names))

    # Check if point cloud is empty
    if len(cloud_data) == 0:
        return None

    # Convert to numpy array
    xyz = [(p[0], p[1], p[2]) for p in cloud_data]
    
    # Create Open3D point cloud
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    
    return o3d_cloud

class ObjectDetectorOpen3D:
    def __init__(self):
        rospy.loginfo("Initializing Plane Object Detector (Open3D)...")
        self.load_params()
        
        self.publisher = rospy.Publisher(self.output_topic, PointStamped, queue_size=10)
        rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("Plane Object Detector (Open3D) is running.")

    def load_params(self):
        self.input_topic = rospy.get_param('~input_topic', '/camera/depth_registered/points')
        self.output_topic = rospy.get_param('~output_topic', '/object_detector/object_position')
        self.voxel_leaf_size = rospy.get_param('~voxel_leaf_size', 0.005) # 5mm
        self.plane_distance_threshold = rospy.get_param('~plane_distance_threshold', 0.015) # 1.5cm
        self.cluster_eps = rospy.get_param('~cluster_eps', 0.03) # 3cm search radius
        self.cluster_min_points = rospy.get_param('~cluster_min_points', 40)

    def cloud_callback(self, ros_cloud):
        # 1. Convert ROS PointCloud2 to Open3D PointCloud
        o3d_cloud = convert_ros_to_o3d(ros_cloud)
        if o3d_cloud is None:
            rospy.logwarn_throttle(10, "Received an empty point cloud.")
            return

        # 2. Voxel Grid Downsampling to speed up processing
        o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=self.voxel_leaf_size)

        # 3. RANSAC Plane Segmentation to find the table
        try:
            plane_model, inliers = o3d_cloud.segment_plane(distance_threshold=self.plane_distance_threshold,
                                                           ransac_n=3,
                                                           num_iterations=1000)
        except Exception as e:
            rospy.logerr(f"Plane segmentation failed: {e}")
            return
        
        # 4. Extract Objects (everything that is NOT the plane)
        object_cloud = o3d_cloud.select_by_index(inliers, invert=True)
        if not object_cloud.has_points():
            rospy.logwarn_throttle(10, "No points found after removing the plane.")
            return

        # 5. Euclidean Clustering using DBSCAN to find distinct objects
        #    Returns a list of labels, where -1 is noise.
        labels = np.array(object_cloud.cluster_dbscan(eps=self.cluster_eps, 
                                                       min_points=self.cluster_min_points, 
                                                       print_progress=False))
        
        max_label = labels.max()
        if max_label < 0:
            rospy.logwarn_throttle(10, "No clusters found matching the criteria.")
            return

        # 6. Find the largest cluster
        # We count the occurrences of each label (excluding noise label -1)
        counts = np.bincount(labels[labels >= 0])
        largest_cluster_label = counts.argmax()
        
        # Select points belonging to the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        largest_cluster_cloud = object_cloud.select_by_index(largest_cluster_indices)

        # 7. Calculate the centroid of the largest cluster
        centroid = largest_cluster_cloud.get_center()

        # 8. Publish the centroid as a PointStamped message
        point_msg = PointStamped()
        point_msg.header = ros_cloud.header # Use original header for frame_id and stamp
        point_msg.point.x = centroid[0]
        point_msg.point.y = centroid[1]
        point_msg.point.z = centroid[2]
        
        self.publisher.publish(point_msg)

if __name__ == '__main__':
    rospy.init_node('plane_object_detector_open3d')
    try:
        ObjectDetectorOpen3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass