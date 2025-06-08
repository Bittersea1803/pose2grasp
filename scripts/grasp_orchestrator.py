#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Pose
from core.real_ur5_controller import UR5Controller

class GraspOrchestrator:
    def __init__(self):
        if UR5Controller is None:
            rospy.signal_shutdown("UR5Controller not available.")
            return
            
        rospy.loginfo("Initializing Grasp Orchestrator...")
        self.load_params()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.reset_state()
        
        rospy.Subscriber(self.grasp_topic, String, self.grasp_callback)
        rospy.Subscriber(self.object_topic, PointStamped, self.object_callback)
        
        rospy.loginfo("Grasp Orchestrator is running and waiting for data.")

    def load_params(self):
        self.required_stability_time = rospy.Duration(rospy.get_param('~stability_time', 2.5))
        self.grasp_topic = rospy.get_param('~grasp_topic', '/pose2grasp/grasp_type')
        self.object_topic = rospy.get_param('~object_topic', '/object_detector/object_position')
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.z_offset_approach = rospy.get_param('~z_offset_approach', 0.15) # 15cm above
        self.z_offset_retreat = rospy.get_param('~z_offset_retreat', 0.20) # 20cm above

    def reset_state(self):
        rospy.loginfo("State has been reset. Ready for a new cycle.")
        self.last_detected_grasp = None
        self.grasp_detection_start_time = None
        self.stable_grasp_type = None
        self.last_known_object_position_cam = None
        self.is_robot_busy = False

    def object_callback(self, msg):
        if not self.is_robot_busy:
            self.last_known_object_position_cam = msg

    def grasp_callback(self, msg):
        if self.is_robot_busy:
            return

        detected_grasp = msg.data
        if detected_grasp != self.last_detected_grasp:
            self.last_detected_grasp = detected_grasp
            self.grasp_detection_start_time = rospy.Time.now()
            self.stable_grasp_type = None
        elif (rospy.Time.now() - self.grasp_detection_start_time) > self.required_stability_time:
            if self.stable_grasp_type != detected_grasp:
                rospy.loginfo(f"Stable grasp detected: {detected_grasp.upper()}")
                self.stable_grasp_type = detected_grasp
                self.check_and_trigger_robot()

    def check_and_trigger_robot(self):
        if self.stable_grasp_type and self.last_known_object_position_cam:
            rospy.loginfo("="*30)
            rospy.loginfo(f"TRIGGER! Conditions met. Starting grasp sequence.")
            rospy.loginfo(f"  -> Grasp: {self.stable_grasp_type}")
            rospy.loginfo(f"  -> Position (Camera Frame): {self.last_known_object_position_cam.point.x:.3f}, {self.last_known_object_position_cam.point.y:.3f}, {self.last_known_object_position_cam.point.z:.3f}")
            rospy.loginfo("="*30)
            
            self.is_robot_busy = True
            self.execute_full_grasp_sequence()
        else:
            rospy.logwarn_throttle(5, "Trigger conditions not yet met. Waiting...")

    def transform_point(self, point_stamped_in):
        try:
            rospy.loginfo(f"Attempting to transform point from '{point_stamped_in.header.frame_id}' to '{self.robot_base_frame}'")
            return self.tf_buffer.transform(point_stamped_in, self.robot_base_frame, rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF2 transform failed: {e}. Is the TF tree being published?")
            return None

    def execute_full_grasp_sequence(self):
        # 1. Transform coordinates
        target_point_robot_frame = self.transform_point(self.last_known_object_position_cam)
        if not target_point_robot_frame:
            self.reset_state()
            return

        target_pose = Pose()
        target_pose.position = target_point_robot_frame.point
        q = quaternion_from_euler(-np.pi, 0, 0)  # -180 deg roll for Z-down
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = q

        controller = UR5Controller()

        pre_grasp_pose = self.create_offset_pose(target_pose, self.z_offset_approach)
        rospy.loginfo("Moving to PRE-GRASP position...")
        if not controller.move_to_pose_goal(pre_grasp_pose):
            rospy.logerr("Failed to reach pre-grasp position. Aborting.")
            self.reset_state()
            return

        rospy.loginfo("TODO: Opening gripper...")
        # controller.control_gripper("open", self.stable_grasp_type)
        rospy.sleep(1.0)

        # Move down to the final grasp position
        rospy.loginfo("Moving to GRASP position...")
        if not controller.move_to_pose_goal(target_pose):
            rospy.logerr("Failed to reach grasp position. Aborting.")
            self.reset_state()
            return

        # -- GRIPPER CONTROL - CLOSE ---
        rospy.loginfo(f"TODO: Closing gripper with '{self.stable_grasp_type}' style...")
        # controller.control_gripper("close", self.stable_grasp_type)
        rospy.sleep(2.0)

        # Move up to a retreat position
        retreat_pose = self.create_offset_pose(target_pose, self.z_offset_retreat)
        rospy.loginfo("Moving to RETREAT position...")
        if not controller.move_to_pose_goal(retreat_pose):
            rospy.logerr("Failed to retreat after grasping. The object might be held.")
            return

        rospy.loginfo("SUCCESS: Grasp sequence completed!")
        
        # controller.move_to_home_position() ?
        
        self.reset_state()


    def create_offset_pose(self, pose, z_offset):
        offset_pose = Pose()
        offset_pose.position.x = pose.position.x
        offset_pose.position.y = pose.position.y
        offset_pose.position.z = pose.position.z + z_offset
        offset_pose.orientation = pose.orientation
        return offset_pose

if __name__ == '__main__':
    rospy.init_node('grasp_orchestrator')
    try:
        orchestrator = GraspOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
