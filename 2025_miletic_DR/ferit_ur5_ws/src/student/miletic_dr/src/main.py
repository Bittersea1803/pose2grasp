#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from core.real_ur5_controller import UR5Controller

class GraspOrchestrator:
    def __init__(self):
        rospy.init_node('grasp_orchestrator_node')
        rospy.loginfo("Initializing Grasp Orchestrator...")

        # --- Konfiguracija ---
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
        
        rospy.loginfo("Orchestrator is in MONITORING_FOR_HAND state.")
        self.move_to_home_pose()

    def move_to_home_pose(self):
        rospy.loginfo("Moving to HOME pose...")
        self.robot_controller.move_to_joint_goal(self.HOME_POSE_JOINTS)
        rospy.loginfo("Robot is at HOME pose. Ready to monitor.")

    def grasp_callback(self, msg):
        if self.state != "MONITORING_FOR_HAND" or self.is_robot_busy:
            return

        detected_grasp = msg.data
        if detected_grasp == "none" or detected_grasp == "--":
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
        
        # 1. Transformiraj pozu objekta iz koordinatnog sustava kamere u sustav robota
        try:
            target_pose_robot_frame = self.tf_buffer.transform(
                self.last_known_object_pose_cam, self.ROBOT_BASE_FRAME, rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(f"TF transform failed: {e}. Aborting grasp.")
            self.reset_system()
            return
            
        rospy.loginfo(f"Object pose in '{self.ROBOT_BASE_FRAME}' frame calculated.")

        # a. Izračunaj prilaznu pozu (10cm iznad objekta)
        pre_grasp_pose = target_pose_robot_frame
        pre_grasp_pose.pose.position.z += 0.10 
        
        # b. Izračunaj pozu za povlačenje (15cm iznad objekta)
        retreat_pose = target_pose_robot_frame
        retreat_pose.pose.position.z += 0.15

        rospy.loginfo("--- Starting grasp execution ---")
        
        # Pomak na prilaznu pozu
        rospy.loginfo("1. Moving to PRE-GRASP pose...")
        self.robot_controller.move_to_pose_goal(pre_grasp_pose)
        rospy.sleep(2.0)

        # Pomak na finalnu pozu za hvatanje
        rospy.loginfo("2. Moving to GRASP pose...")
        self.robot_controller.move_to_pose_goal(target_pose_robot_frame)
        rospy.sleep(2.0)

        # Zatvaranje hvataljke
        rospy.loginfo(f"3. Closing gripper with style: '{self.stable_grasp_type}'")
        self.robot_controller.send_gripper_command(self.stable_grasp_type, ...)
        rospy.sleep(1.5)

        # Pomak na pozu za povlačenje
        rospy.loginfo("4. Moving to RETREAT pose...")
        self.robot_controller.move_to_pose_goal(retreat_pose)
        rospy.sleep(2.0)
        
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
        orchestrator = GraspOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass