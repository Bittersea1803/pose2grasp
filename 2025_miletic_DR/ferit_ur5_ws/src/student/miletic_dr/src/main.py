#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from core.real_ur5_controller import UR5Controller
from core.transforms import pose_to_matrix

class GraspOrchestrator:
    def __init__(self):
        rospy.init_node('main_node')
        rospy.loginfo("Initializing main node...")

        self.GRASP_TOPIC = "/pose2grasp/grasp_type"
        self.OBJECT_TOPIC = "/plane_object_detector/biggest_object_position"
        self.ROBOT_BASE_FRAME = "base_link"
        self.CAMERA_FRAME = "camera_rgb_optical_frame"
        self.REQUIRED_STABILITY_TIME_SEC = rospy.get_param('~stability_time', 2.0)

        self.HOME_POSE_JOINTS = np.deg2rad([0, -90, 90, -90, -90, 0])
        self.SCAN_POSE_JOINTS = np.deg2rad([0, -70, 110, -130, -90, 0])

        self.robot_controller = UR5Controller()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.state = "MONITORING_FOR_HAND"
        self.last_detected_grasp = None
        self.grasp_detection_start_time = None
        self.stable_grasp_type = None
        self.last_known_object_pose_cam = None
        self.is_robot_busy = False

        self.grasp_sub = rospy.Subscriber(self.GRASP_TOPIC, String, self.grasp_callback)
        self.object_sub = None

        rospy.loginfo("main is in MONITORING_FOR_HAND state.")
        self.move_to_home_pose()

    def move_to_home_pose(self):
        rospy.loginfo("Moving to HOME pose...")
        self.robot_controller.move_to_joint_goal(self.HOME_POSE_JOINTS)
        rospy.loginfo("Robot is at HOME pose. Ready to monitor.")

    def grasp_callback(self, msg):
        if self.state != "MONITORING_FOR_HAND" or self.is_robot_busy:
            return

        detected_grasp = msg.data
        if detected_grasp in ["none", "--"]:
            self.last_detected_grasp = None
            return

        if detected_grasp != self.last_detected_grasp:
            self.last_detected_grasp = detected_grasp
            self.grasp_detection_start_time = rospy.Time.now()
        elif (rospy.Time.now() - self.grasp_detection_start_time).to_sec() > self.REQUIRED_STABILITY_TIME_SEC:
            rospy.loginfo(f"Stable grasp '{detected_grasp}' detected!")
            self.stable_grasp_type = detected_grasp
            self.is_robot_busy = True
            self.state = "MOVING_TO_SCAN_POSE"
            self.move_to_scan_pose()

    def move_to_scan_pose(self):
        rospy.loginfo("State change: MOVING_TO_SCAN_POSE")
        rospy.loginfo("Moving robot to scan for objects...")

        self.robot_controller.move_to_joint_goal(self.SCAN_POSE_JOINTS)

        rospy.loginfo("Robot is in SCAN pose.")
        self.state = "SCANNING_FOR_OBJECT"
        self.object_sub = rospy.Subscriber(self.OBJECT_TOPIC, PoseStamped, self.object_callback)
        rospy.loginfo("State change: SCANNING_FOR_OBJECT. Waiting for object detection...")

    def object_callback(self, msg):
        if self.state != "SCANNING_FOR_OBJECT":
            return

        rospy.loginfo("Object detected!")
        self.last_known_object_pose_cam = msg

        if self.object_sub:
            self.object_sub.unregister()
            self.object_sub = None

        self.state = "EXECUTING_GRASP"
        self.execute_full_grasp_sequence()

    def execute_full_grasp_sequence(self):
        rospy.loginfo("State change: EXECUTING_GRASP")

        try:
            target_pose_robot_frame = self.tf_buffer.transform(
                self.last_known_object_pose_cam, self.ROBOT_BASE_FRAME, rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(f"TF transform failed: {e}. Aborting grasp.")
            self.reset_system()
            return

        rospy.loginfo(f"Object pose in '{self.ROBOT_BASE_FRAME}' frame calculated.")

        # Define approach and retreat poses
        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header = target_pose_robot_frame.header
        pre_grasp_pose.pose = target_pose_robot_frame.pose
        pre_grasp_pose.pose.position.z += 0.10

        retreat_pose = PoseStamped()
        retreat_pose.header = target_pose_robot_frame.header
        retreat_pose.pose = target_pose_robot_frame.pose
        retreat_pose.pose.position.z += 0.15

        rospy.loginfo("--- Starting grasp execution ---")

        for name, pose in [("PRE-GRASP", pre_grasp_pose),
                           ("GRASP", target_pose_robot_frame),
                           ("RETREAT", retreat_pose)]:
            pose_matrix = pose_to_matrix(pose.pose)
            joint_goal = self.robot_controller.get_closest_ik_solution(pose_matrix)
            if joint_goal is not None:
                rospy.loginfo(f"Moving to {name} pose...")
                self.robot_controller.move_to_joint_goal(joint_goal)
                rospy.sleep(2.0)
            else:
                rospy.logerr(f"No IK solution found for {name} pose. Aborting.")
                self.reset_system()
                return

        rospy.loginfo(f"Closing gripper with grasp: '{self.stable_grasp_type}'")
        self.robot_controller.send_gripper_command(self.stable_grasp_type, ...)
        rospy.sleep(1.5)

        rospy.loginfo("--- Grasp sequence finished! ---")
        self.reset_system()

    def reset_system(self):
        self.move_to_home_pose()
        self.state = "MONITORING_FOR_HAND"
        self.last_detected_grasp = None
        self.grasp_detection_start_time = None
        self.stable_grasp_type = None
        self.last_known_object_pose_cam = None
        self.is_robot_busy = False

        if self.object_sub:
            self.object_sub.unregister()
            self.object_sub = None

        rospy.loginfo("System reset. State: MONITORING_FOR_HAND")


if __name__ == '__main__':
    try:
        grasp_orchestrator_instance = GraspOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
