import os
import sys
import threading
import numpy as np
import cv2
import rospy
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class PlaneObjectDetector:
    def __init__(self):
        rospy.init_node("plane_object_detector_node", anonymous=True)
        self.bridge = CvBridge()
        self.is_shutdown = False

        # Intrinsic camera parameters (to be filled by CameraInfo)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.current_frame_id = None # To store the frame_id for publishing
        self.camera_info_received = False

        # ROS topic names (adjust if needed)
        self.RGB_TOPIC = "/camera/rgb/image_rect_color"
        self.DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
        self.INFO_TOPIC = "/camera/rgb/camera_info"
        self.OBJECT_OUTPUT_TOPIC = "/plane_object_detector/biggest_object_position" # Topic for the biggest object

        self.object_pos_publisher = rospy.Publisher(self.OBJECT_OUTPUT_TOPIC, PointStamped, queue_size=1)

        # Set up synchronized subscribers
        self._setup_subscribers()

        # Initialize Open3D visualizer
        self._init_open3d_visualizer()

        # Start ROS spin in separate thread
        threading.Thread(target=self._ros_spin, daemon=True).start()
        rospy.loginfo("[PlaneObjectDetector] Node initialized, awaiting data...")

    def _setup_subscribers(self):
        rgb_sub = message_filters.Subscriber(self.RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(self.DEPTH_TOPIC, Image)
        info_sub = message_filters.Subscriber(self.INFO_TOPIC, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self._callback_synced)
        rospy.loginfo(f"[PlaneObjectDetector] Subscribed to:\n  RGB: {self.RGB_TOPIC}\n  DEPTH: {self.DEPTH_TOPIC}\n  INFO: {self.INFO_TOPIC}")

    def _ros_spin(self):
        rospy.spin()
        rospy.loginfo("[PlaneObjectDetector] ROS spin thread ended.")

    def _callback_synced(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self.is_shutdown:
            return

        # Get camera intrinsics once
        if not self.camera_info_received:
            K = info_msg.K
            if len(K) == 9 and K[0] > 0:
                self.fx, self.fy = K[0], K[4]
                self.cx, self.cy = K[2], K[5]
                self.current_frame_id = depth_msg.header.frame_id # Store frame_id
                self.camera_info_received = True
                rospy.loginfo(f"[PlaneObjectDetector] Got camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            else:
                rospy.logwarn_throttle(5, "[PlaneObjectDetector] Invalid CameraInfo, skipping frame.")
                return

        # Convert ROS images to OpenCV
        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"[PlaneObjectDetector] CvBridge Error: {e}")
            return

        # Process the frame: build point cloud, segment plane, cluster objects, visualize
        self._process_frame(rgb_cv, depth_cv)

    def _process_frame(self, rgb_bgr: np.ndarray, depth_16u: np.ndarray):
        # Convert depth (16UC1) to point cloud (Open3D format)
        pcd = self._create_point_cloud_from_depth(depth_16u, rgb_bgr)
        if pcd is None:
            return

        # 1) Downsample for speed
        voxel_size = 0.005  # 5mm
        pcd_down = pcd.voxel_down_sample(voxel_size)

        # 2) Estimate normals (needed for plane segmentation)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

        # 3) Segment dominant plane using RANSAC
        plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=1000)
        [a, b, c, d] = plane_model
        rospy.loginfo_throttle(2, f"[PlaneObjectDetector] Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Extract plane and objects point clouds
        plane_cloud = pcd_down.select_by_index(inliers)
        plane_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # light gray for plane
        objects_cloud = pcd_down.select_by_index(inliers, invert=True)
        objects_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # red for all objects

        # 4) Further filter objects: keep only points a certain height above plane
        #    (to remove small noise near plane). Compute plane distances of each object point.
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)
        plane_d = d
        points_objects = np.asarray(objects_cloud.points)
        # Distance from plane for each point: (n . x + d) / ||n||
        distances = (points_objects @ plane_normal + plane_d)
        # Consider only points with positive distance > threshold (e.g., 2cm)
        height_thresh = 0.02
        mask_above = distances > height_thresh

        # Create a point cloud of objects above the plane
        filtered_objects_pcd = o3d.geometry.PointCloud()
        if np.count_nonzero(mask_above) >= 10: # Need at least some points for further processing
            filtered_objects_pcd.points = o3d.utility.Vector3dVector(points_objects[mask_above])
            # Color these points for visualization (e.g., green)
            # filtered_objects_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Optional: if objects_cloud was not colored
        else:
            rospy.loginfo_throttle(5, "[PlaneObjectDetector] Not enough points above plane to detect objects.")
            self._update_open3d_visualization(plane_cloud, None, []) # Pass empty list for bboxes
            return

        if not filtered_objects_pcd.has_points(): # Should be caught by above, but as a safeguard
            self._update_open3d_visualization(plane_cloud, None, [])
            return

        # 5) Cluster the filtered object points using DBSCAN
        eps = 0.02  # 2cm
        min_points = 20
        labels = np.array(filtered_objects_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        # If no labels or all points are noise (max_label = -1)
        if labels.size == 0 or labels.max() < 0:
            rospy.loginfo_throttle(5, "[PlaneObjectDetector] DBSCAN found no clusters or only noise.")
            self._update_open3d_visualization(plane_cloud, filtered_objects_pcd, []) # Show plane and unclustered points
            return

        max_label = labels.max()
        rospy.loginfo_throttle(2, f"[PlaneObjectDetector] DBSCAN initially found {max_label + 1} potential clusters.")

        # 6) For each valid cluster, compute oriented bounding box and collect data
        valid_clusters_data = [] # List of dicts: {'id', 'pcd', 'obb', 'num_points'}
        for cluster_id in range(max_label + 1):
            mask = (labels == cluster_id)
            num_points_in_cluster = np.count_nonzero(mask)

            if num_points_in_cluster < min_points: # Filter out clusters smaller than DBSCAN's min_points
                continue

            cluster_pts = np.asarray(filtered_objects_pcd.points)[mask]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
            # Normals are not strictly required for OBB but can be estimated if needed for other purposes
            # cluster_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            obb = cluster_pcd.get_oriented_bounding_box()
            
            # Color OBB and cluster points for visualization
            color = np.random.rand(3)
            obb.color = color
            cluster_pcd.paint_uniform_color(color)
            valid_clusters_data.append({
                'id': cluster_id,
                'pcd': cluster_pcd, # This pcd is for visualization of the cluster itself
                'obb': obb,
                'num_points': num_points_in_cluster
            })

        # 7) Publish position of the biggest object and update visualization
        cluster_geoms_for_viz = [(data['pcd'], data['obb']) for data in valid_clusters_data]

        if valid_clusters_data:
            valid_clusters_data.sort(key=lambda c: c['num_points'], reverse=True) # Sort by num_points
            biggest_object_data = valid_clusters_data[0]
            biggest_object_center = biggest_object_data['obb'].get_center()

            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now() # Use current time for published message
            point_msg.header.frame_id = self.current_frame_id
            point_msg.point.x, point_msg.point.y, point_msg.point.z = biggest_object_center
            self.object_pos_publisher.publish(point_msg)
            rospy.loginfo_throttle(1, f"[PlaneObjectDetector] Published biggest object center: [{point_msg.point.x:.3f}, {point_msg.point.y:.3f}, {point_msg.point.z:.3f}] in frame {self.current_frame_id} ({biggest_object_data['num_points']} points)")
        else:
            rospy.loginfo_throttle(2, "[PlaneObjectDetector] No sufficiently large object clusters found to publish.")

        self._update_open3d_visualization(plane_cloud, filtered_objects_pcd, cluster_geoms_for_viz)

    def _create_point_cloud_from_depth(self, depth_16u: np.ndarray, rgb_bgr: np.ndarray):
        """
        Convert depth image and aligned RGB into an Open3D point cloud.
        Assumes depth is in millimeters.
        """
        if not self.camera_info_received:
            return None

        H, W = depth_16u.shape
        # Create a mask of valid depth
        valid_mask = (depth_16u > 400) & (depth_16u < 1500)  # in mm
        indices = np.array(np.nonzero(valid_mask)).T  # (N,2)
        if indices.shape[0] == 0:
            return None

        # Build arrays of u,v coordinates and depth in meters
        us = indices[:, 1].astype(np.float32)
        vs = indices[:, 0].astype(np.float32)
        zs = depth_16u[indices[:, 0], indices[:, 1]].astype(np.float32) / 1000.0  # convert to meters

        # Compute X, Y, Z
        xs = (us - self.cx) * zs / self.fx
        ys = (vs - self.cy) * zs / self.fy
        points = np.stack((xs, ys, zs), axis=-1)

        # Color for each point
        rgb_resized = cv2.resize(rgb_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
        colors = rgb_resized[indices[:, 0], indices[:, 1], :] / 255.0  # normalize to [0,1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _init_open3d_visualizer(self):
        """
        Initialize the Open3D Visualizer and geometries.
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Plane and Object Detection", width=800, height=600)

        # Geometry for plane point cloud
        self.plane_geom = o3d.geometry.PointCloud()
        # Geometry for object points (all)
        self.objects_geom = o3d.geometry.PointCloud()
        # Geometry for clustering: not used directly; bounding boxes added separately

        # Add to visualizer
        self.vis.add_geometry(self.plane_geom)
        self.vis.add_geometry(self.objects_geom)

        # Keep track of bounding box geometries (LineSets)
        self.bbox_geoms = []

        # Run visualizer in separate thread
        threading.Thread(target=self._open3d_run, daemon=True).start()

    def _open3d_run(self):
        """
        Continuously run the Open3D event loop until shutdown.
        """
        while not self.is_shutdown and self.vis.poll_events():
            self.vis.update_renderer()
        # On window close, shut down
        self.is_shutdown = True
        rospy.signal_shutdown("Open3D window closed by user.")
        rospy.loginfo("[PlaneObjectDetector] Open3D window closed, shutting down.")

    def _update_open3d_visualization(self, plane_cloud, objects_cloud, cluster_bboxes):
        """
        Update the geometries in the Open3D visualizer:
          - plane_cloud: PointCloud of plane in gray
          - objects_cloud: PointCloud of all objects (red)
          - cluster_bboxes: list of tuples (cluster_pcd, obb) to visualize
        """
        # Clear existing bounding boxes
        for geom in self.bbox_geoms:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.bbox_geoms = []

        # Update plane geometry
        if plane_cloud is not None:
            self.plane_geom.points = plane_cloud.points
            self.plane_geom.colors = plane_cloud.colors
            self.vis.update_geometry(self.plane_geom)
        
        # Update object geometry
        if objects_cloud is not None:
            self.objects_geom.points = objects_cloud.points
            self.objects_geom.colors = objects_cloud.colors
            self.vis.update_geometry(self.objects_geom)
        
        # Add each cluster's bounding box to the visualizer
        if cluster_bboxes is not None:
            for (cluster_pcd, obb) in cluster_bboxes:
                # Visualize cluster points as faint points, if desired
                # Alternatively, skip adding cluster_pcd and only show bounding boxes
                # o3d.geometry.PointCloud.paint_uniform_color(cluster_pcd, cluster_pcd.colors[0])
                # self.vis.add_geometry(cluster_pcd, reset_bounding_box=False)

                # Draw oriented bounding box as LineSet
                lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
                self.bbox_geoms.append(lineset)
                self.vis.add_geometry(lineset, reset_bounding_box=False)

        # Trigger render update
        self.vis.update_renderer()
        rospy.loginfo_throttle(1, "[PlaneObjectDetector] Visualization updated.")
        
    def shutdown(self):
        self.is_shutdown = True
        try:
            self.vis.destroy_window()
        except Exception:
            pass
        rospy.loginfo("[PlaneObjectDetector] Shutting down.")


if __name__ == "__main__":
    try:
        node = PlaneObjectDetector()
        rospy.loginfo("[PlaneObjectDetector] Running... Press Ctrl+C to exit.")

        while not rospy.is_shutdown() and not node.is_shutdown:
            rospy.sleep(0.1)

        node.shutdown()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("[PlaneObjectDetector] Node terminated.")