#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import sys
from tf.transformations import quaternion_matrix, euler_from_matrix

import rospkg

from std_msgs.msg import String
from geometry_msgs.msg import Pose
import roslib

from miletic_dr.msg import DetectedObject 

from core.real_ur5_controller import UR5Controller

class GraspOrchestrator:
    def __init__(self):
        rospy.init_node('main_node')
        rospy.loginfo("Initializing Grasp Orchestrator...")

        self.POSE_TOPIC = "/miletic_dr/pose_type"
        self.OBJECT_TOPIC = "/miletic_dr/detected_object"
        
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('miletic_dr')
            config_path = os.path.join(package_path, 'config')
            
            self.T_C_6 = np.load(os.path.join(config_path, 'T_C_6.npy'))
            self.T_G_6 = np.load(os.path.join(config_path, 'T_G_6.npy'))

            rospy.loginfo("Transformation matrices loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load transformation matrices from config folder: {e}")
            rospy.logfatal("Please ensure T_C_6.npy and T_G_6.npy exist in 'miletic_dr/config/'.")
            sys.exit(1)

        self.robot_controller = UR5Controller()

        self.HOME_POSE_JOINTS = np.deg2rad([-89, -6, -140, -54, 91, 45])
        self.SCAN_POSE_JOINTS = np.deg2rad([-91.13, -98.58, -22.66, -143.08, 92.83, 45.45])

        self.state = "MONITORING_FOR_HAND"
        
        # --- PROMJENA OVDJE ---
        # Uklonjene varijable za provjeru stabilnosti koje više nisu potrebne
        # self.last_detected_pose = None
        # self.pose_detection_start_time = None
        self.stable_pose_type = None
        self.is_robot_busy = False
        
        self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, String, self.pose_callback)
        self.object_sub = None

        rospy.loginfo("main is in MONITORING_FOR_HAND state.")
        rospy.Timer(rospy.Duration(1), self.initial_move_callback, oneshot=True)

    def initial_move_callback(self, event):
        rospy.loginfo("Timer triggered: Moving to HOME pose for the first time.")
        self.move_to_home_pose()

    def move_to_home_pose(self):
        rospy.loginfo("Moving to HOME pose...")
        self.is_robot_busy = True
        
        joint_trajectory_points = np.array([
            self.robot_controller.get_current_joint_values(), 
            self.HOME_POSE_JOINTS
        ])
        self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)
        
        self.is_robot_busy = False
        rospy.loginfo("Robot is at HOME pose. Ready to monitor.")

    # --- GLAVNA PROMJENA JE OVDJE ---
    def pose_callback(self, msg):
        """
        Ova funkcija sada reagira odmah na prvu primljenu pozu
        koja nije 'none'.
        """
        rospy.loginfo(f"Received pose type: {msg.data}")
        
        # Zanemari poruke ako robot već nešto radi ili nije u stanju čekanja
        if self.state != "MONITORING_FOR_HAND" or self.is_robot_busy:
            return

        detected_pose = msg.data.lower()
        
        # Zanemari "none" ili prazne poruke
        if detected_pose in ["none", "--", None, ""]:
            return

        # Ako smo primili bilo koju valjanu pozu (npr. 'basic', 'pinch')
        # i u stanju smo praćenja, odmah pokrećemo akciju.
        rospy.loginfo(f"Pose '{detected_pose}' detected! Proceeding immediately.")
        
        self.stable_pose_type = detected_pose  # Spremi tip detektirane poze
        self.is_robot_busy = True              # Označi robota kao zauzetog
        self.state = "MOVING_TO_SCAN_POSE"     # Promijeni stanje
        self.move_to_scan_pose()               # Pozovi funkciju za pomicanje robota

    def move_to_scan_pose(self):
        rospy.loginfo("State change: MOVING_TO_SCAN_POSE")
        joint_trajectory_points = np.array([
            self.robot_controller.get_current_joint_values(), 
            self.SCAN_POSE_JOINTS
        ])

        self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)
        rospy.loginfo("Robot is in SCAN pose.")
        self.state = "SCANNING_FOR_OBJECT"
        self.object_sub = rospy.Subscriber(self.OBJECT_TOPIC, DetectedObject, self.object_callback)
        rospy.loginfo("State change: SCANNING_FOR_OBJECT. Waiting for object detection...")

    def object_callback(self, msg: DetectedObject):
        if self.state != "SCANNING_FOR_OBJECT":
            return

        rospy.loginfo("Detected object data received!")
        if self.object_sub:
            self.object_sub.unregister(); self.object_sub = None
        
        object_pose_in_cam = msg.pose
        object_dims_in_cam = msg.dimensions
        
        self.state = "EXECUTING_GRASP"
        self.execute_full_grasp_sequence(object_pose_in_cam, object_dims_in_cam)

    def pose_msg_to_matrix(self, pose_msg: Pose):
        q = pose_msg.orientation
        p = pose_msg.position
        
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = p.x
        T[1, 3] = p.y
        T[2, 3] = p.z
        return T

    def calculate_grasp_pose(self, T_B_O, dims, grasp_type):
        rospy.loginfo(f"Calculating grasp pose for type: '{grasp_type}'")
        
        target_position_in_base = T_B_O[:3, 3].copy()
        
        if grasp_type.lower() == "scissor":
            object_x_axis_in_base = T_B_O[:3, 0]
            object_width = dims.x
            offset_vector = object_x_axis_in_base * (object_width / 2.0)
            target_position_in_base += offset_vector
            rospy.loginfo(f"Applying offset for scissor grasp: {np.round(offset_vector, 3)}")

        R_O_in_B = T_B_O[:3, :3]
        gripper_z = -R_O_in_B[:, 2] 
        gripper_y = R_O_in_B[:, 1]
        gripper_x = np.cross(gripper_y, gripper_z)
        R_G_in_B = np.stack([gripper_x, gripper_y, gripper_z], axis=1)

        T_B_G = np.identity(4)
        T_B_G[:3, :3] = R_G_in_B
        T_B_G[:3, 3] = target_position_in_base
        
        return T_B_G #T

    def execute_full_grasp_sequence(self, object_pose_cam, object_dims_cam):
        rospy.loginfo("State change: EXECUTING_GRASP")

        T_B_6 = self.robot_controller.get_current_tool_pose()
        T_C_O = self.pose_msg_to_matrix(object_pose_cam)
        T_B_O = T_B_6 @ self.T_C_6 @ T_C_O
        T_B_G_target = self.calculate_grasp_pose(T_B_O, object_dims_cam, self.stable_pose_type) # eqals T B TCP
        T_B_TCP = T_B_G_target


        #TODO: imepltnirat
        T B 6_= T B TCP * T TCP 6



        T_B_T_target = T_B_G_target @ np.linalg.inv(self.T_G_6)
        T_B_T_pregrasp = T_B_T_target.copy()
        T_B_T_pregrasp[2, 3] += 0.12
        T_B_T_retreat = T_B_T_target.copy()
        T_B_T_retreat[2, 3] += 0.15

        rospy.loginfo("--- Starting grasp execution sequence ---")

        self.robot_controller.send_gripper_command(self.stable_pose_type, position=0)
        if not self.move_to_matrix_goal(T_B_T_pregrasp, "PRE-GRASP"): return
        if not self.move_to_matrix_goal(T_B_T_target, "GRASP"): return
        self.robot_controller.send_gripper_command(self.stable_pose_type, position=255)
        if not self.move_to_matrix_goal(T_B_T_retreat, "RETREAT"): return

        rospy.loginfo("--- Grasp sequence finished successfully! ---")
        self.reset_system()

    def move_to_matrix_goal(self, pose_matrix, name):
        joint_goal = self.robot_controller.get_closest_ik_solution(pose_matrix)
        if joint_goal is not None:
            rospy.loginfo(f"Moving to {name} pose...")
            joint_trajectory_points = np.array([
                self.robot_controller.get_current_joint_values(), 
                joint_goal
            ])
            self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)
            return True
        else:
            rospy.logerr(f"No IK solution for {name} pose. Aborting grasp sequence.")
            self.reset_system()
            return False
        
        # robotu za inverznu predajmo t 6 b, ne pradjem mu pregrasp neg

    def reset_system(self):
        self.move_to_home_pose()
        self.state = "MONITORING_FOR_HAND"
        if self.object_sub:
            self.object_sub.unregister(); self.object_sub = None
        rospy.loginfo("System reset. State: MONITORING_FOR_HAND")


if __name__ == '__main__':
    try:
        grasp_orchestrator = GraspOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass