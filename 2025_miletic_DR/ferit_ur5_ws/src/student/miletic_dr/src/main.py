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

from miletic_dr.msg import DetectedObject 

from core.real_ur5_controller import UR5Controller

class GraspOrchestrator:
    def __init__(self):
        rospy.init_node('main_node')
        rospy.loginfo("Initializing Grasp Orchestrator...")

        self.GRASP_TOPIC = "/miletic_dr/pose_type"
        self.OBJECT_TOPIC = "/miletic_dr/detected_object"
        
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('miletic_dr')
            config_path = os.path.join(package_path, 'config')
            
            # T_C_6: Transformacija od kamere do 6. zgloba robota
            self.T_C_6 = np.load(os.path.join(config_path, 'T_C_6.npy'))
            # T_G_6: Transformacija od hvataljke do 6. zgloba robota
            self.T_G_6 = np.load(os.path.join(config_path, 'T_G_6.npy'))

            rospy.loginfo("Transformation matrices loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load transformation matrices from config folder: {e}")
            rospy.logfatal("Please ensure T_C_6.npy and T_G_6.npy exist in 'miletic_dr/config/'.")
            sys.exit(1)

        self.robot_controller = UR5Controller()
        
        # TODO: Odrediti
        self.HOME_POSE_JOINTS = np.deg2rad([0, -90, 90, -90, -90, 0])
        self.SCAN_POSE_JOINTS = np.deg2rad([0, -70, 110, -130, -90, 0])

        self.state = "MONITORING_FOR_HAND"
        self.last_detected_grasp = None
        self.grasp_detection_start_time = None
        self.stable_grasp_type = None
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

        detected_grasp = msg.data.lower()
        if detected_grasp in ["none", "--", None, ""]:
            self.last_detected_grasp = None
            return

        REQUIRED_STABILITY_TIME_SEC = 2.0
        if detected_grasp != self.last_detected_grasp:
            self.last_detected_grasp = detected_grasp
            self.grasp_detection_start_time = rospy.Time.now()
        elif (rospy.Time.now() - self.grasp_detection_start_time).to_sec() > REQUIRED_STABILITY_TIME_SEC:
            rospy.loginfo(f"Stable grasp '{detected_grasp}' detected!")
            self.stable_grasp_type = detected_grasp
            self.is_robot_busy = True
            self.state = "MOVING_TO_SCAN_POSE"
            self.move_to_scan_pose()

    def move_to_scan_pose(self):
        rospy.loginfo("State change: MOVING_TO_SCAN_POSE")
        self.robot_controller.move_to_joint_goal(self.SCAN_POSE_JOINTS)
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
        
        # target je centar objekta
        target_position_in_base = T_B_O[:3, 3].copy()
        
        # Pomak za pinch i scissor
        if grasp_type.lower() in ["pinch", "scissor"]:
            object_x_axis_in_base = T_B_O[:3, 0]
            object_width = dims.x
            
            # pomakni ciljnu točku od centra za pola širine duž kraće osi
            offset_vector = object_x_axis_in_base * (object_width / 2.0)
            target_position_in_base += offset_vector
            rospy.loginfo(f"Applying offset for pinch/scissor grasp: {np.round(offset_vector, 3)}")

        # Orijentacija hvataljke
        R_O_in_B = T_B_O[:3, :3]
        gripper_z = -R_O_in_B[:, 2] 
        gripper_y = R_O_in_B[:, 1]
        gripper_x = np.cross(gripper_y, gripper_z)
        R_G_in_B = np.stack([gripper_x, gripper_y, gripper_z], axis=1)

        # finalnu 4x4 transformacijsku matricu za hvataljku
        T_B_G = np.identity(4)
        T_B_G[:3, :3] = R_G_in_B
        T_B_G[:3, 3] = target_position_in_base
        
        return T_B_G

    def execute_full_grasp_sequence(self, object_pose_cam, object_dims_cam):
        rospy.loginfo("State change: EXECUTING_GRASP")

        # 1. Trenutna poza 6. zgloba u odnosu na bazu robota
        T_B_6 = self.robot_controller.get_current_tool_pose()
        # 2. Poza objekta u odnosu na kameru
        T_C_O = self.pose_msg_to_matrix(object_pose_cam)
        
        # 3. Fin: poza objekta u odnosu na bazu robota
        # T_B_O = T_B_6 * T_6_C * T_C_O
        T_B_O = T_B_6 @ np.linalg.inv(self.T_C_6) @ T_C_O

        # poza hvatanja
        T_B_G_target = self.calculate_grasp_pose(T_B_O, object_dims_cam, self.stable_grasp_type)

        # Od poze hvataljke (G) računamo pozu alata (T) koju šaljemo robotu
        # T_B_T = T_B_G * T_G_T^-1
        T_B_T_target = T_B_G_target @ np.linalg.inv(self.T_G_6)

        # Definiraj prilaznu i točku podizanja
        T_B_T_pregrasp = T_B_T_target.copy()
        T_B_T_pregrasp[2, 3] += 0.12  # Prilazak 12cm iznad

        T_B_T_retreat = T_B_T_target.copy()
        T_B_T_retreat[2, 3] += 0.15 # Podizanje 15cm iznad

        rospy.loginfo("--- Starting grasp execution sequence ---")

        # 1. Otvori hvataljku
        self.robot_controller.send_gripper_command(self.stable_grasp_type, position=0)

        # 2. Idi na prilaznu točku
        if not self.move_to_matrix_goal(T_B_T_pregrasp, "PRE-GRASP"): return

        # 3. Idi na točku hvata
        if not self.move_to_matrix_goal(T_B_T_target, "GRASP"): return
        
        # 4. Zatvori hvataljku
        self.robot_controller.send_gripper_command(self.stable_grasp_type, position=255)
        
        # 5. Idi na točku podizanja
        if not self.move_to_matrix_goal(T_B_T_retreat, "RETREAT"): return

        rospy.loginfo("--- Grasp sequence finished successfully! ---")
        self.reset_system()

    def move_to_matrix_goal(self, pose_matrix, name):
        joint_goal = self.robot_controller.get_closest_ik_solution(pose_matrix)
        if joint_goal is not None:
            rospy.loginfo(f"Moving to {name} pose...")
            self.robot_controller.move_to_joint_goal(joint_goal)
            rospy.sleep(1.0)
            return True
        else:
            rospy.logerr(f"No IK solution for {name} pose. Aborting grasp sequence.")
            self.reset_system()
            return False

    def reset_system(self):
        self.move_to_home_pose()
        self.state = "MONITORING_FOR_HAND"
        self.last_detected_grasp = None
        self.is_robot_busy = False
        if self.object_sub:
            self.object_sub.unregister(); self.object_sub = None
        rospy.loginfo("System reset. State: MONITORING_FOR_HAND")


if __name__ == '__main__':
    try:
        grasp_orchestrator = GraspOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
