#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import sys
import json
import rospkg
from tf.transformations import quaternion_matrix

from core.real_ur5_controller import UR5Controller


POSE_FILENAME = "pose_result.txt"
OBJECT_FILENAME = "object_data.json"

HOME_POSE_JOINTS = np.deg2rad([-89, -6, -140, -54, 91, 45])
APPROACH_DISTANCE = 0.12

class GraspExecutor:
    def __init__(self):
        rospy.init_node('grasp_executor_node', anonymous=True)
        rospy.loginfo("Initializing Grasp Executor Node...")
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('miletic_dr')
            config_path = os.path.join(package_path, 'config')
            self.T_C_6 = np.load(os.path.join(config_path, 'T_C_6.npy'))
            self.T_TCP_6 = np.load(os.path.join(config_path, 'T_TCP_6.npy'))
            rospy.loginfo("Transformation matrices T_C_6 and T_TCP_6 loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load transformation matrices: {e}")
            sys.exit(1)
        self.robot_controller = UR5Controller()
        rospy.loginfo("UR5 Controller initialized.")
        # self.robot_controller.activate_gripper()

    def run(self):
        rospy.loginfo("--- Starting Grasp Execution Sequence ---")
        rospy.loginfo("Step 1: Loading data from input files...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pose_type = self._load_pose_file(os.path.join(script_dir, POSE_FILENAME))
        object_data = self._load_object_file(os.path.join(script_dir, OBJECT_FILENAME))
        if not pose_type or not object_data:
            rospy.logfatal("Failed to load input files. Aborting.")
            return
        rospy.loginfo("Step 2: Calculating grasp poses.")
        poses = self._calculate_poses(object_data, pose_type)
        if not poses:
            rospy.logerr("Failed to calculate grasp poses, aborting.")
            return
        rospy.loginfo("Step 3: Executing the grasping motion.")
        self._execute_grasp_motion(poses, pose_type)
        rospy.loginfo("Step 4: Grasp attempt finished. Moving back to HOME pose.")
        self.move_to_home_pose()
        rospy.loginfo("--- Grasp Execution Sequence Complete ---")
        rospy.signal_shutdown("Task complete.")

    def _calculate_poses(self, object_data, pose_type):
        rospy.loginfo("Calculating transformations...")
        
        # Object's pose relative to the camera frame
        T_C_O = np.identity(4)
        T_C_O[:3, :3] = np.array(object_data['rotation_matrix'])
        T_C_O[:3, 3] = np.array(object_data['center'])

        T_B_6_scan = np.array(object_data['T_B_6_scan'])
        
        # T_Base_Object = T_Base_Flange_Scan @  @ T_Camera_Object
        T_B_Object = T_B_6_scan @ self.T_C_6 @ T_C_O
        
        rospy.loginfo("Calculated T_B_O (Object in Base Frame).")

        T_B_TCP = self._calculate_target_tcp_pose(T_B_Object, np.array(object_data['dimensions']), pose_type)
        
        T_B_6 = T_B_TCP @ np.linalg.inv(self.T_TCP_6)
        rospy.loginfo("Calculated T_B_6 (target flange pose) from T_B_TCP.")

        rospy.loginfo("Grasp pose calculated. Skipping approach/retreat poses.")
        return {"grasp": T_B_6}
    
    def _execute_grasp_motion(self, poses, pose_type):
        rospy.loginfo("--- Starting physical grasp motion ---")
        # self.robot_controller.send_gripper_command(pose_type, position=0) # Open gripper
        
        rospy.loginfo("Moving directly to GRASP pose.")
        if not self._move_to_goal(poses["grasp"], "GRASP"): 
            rospy.logerr("Failed to move to GRASP pose.")
            return

        # self.robot_controller.send_gripper_command(pose_type, position=255) # Close gripper
        rospy.sleep(2.0)


        T_B_6_lift = poses["grasp"].copy()
        T_B_6_lift[2, 3] += 0.15 # Z + 10
        self._move_to_goal(T_B_6_lift, "LIFT")

        rospy.loginfo("--- Grasp motion sequence finished successfully! ---")

    def _calculate_target_tcp_pose(self, T_B_O, dims, pose_type):
        R_B_O = T_B_O[:3, :3]
        target_position = T_B_O[:3, 3].copy()
        
        if pose_type.lower() == "scissor":
            rospy.loginfo("Applying positional offset for Scissor Grasp to target the object's edge.")
            object_x_axis_in_base = R_B_O[:, 0]
            object_x_dimension = dims[0]
            offset_vector = object_x_axis_in_base * (object_x_dimension / 2.0)
            target_position -= offset_vector
            
        rospy.loginfo("Calculating orientation for the gripper.")
        gripper_z_axis = -R_B_O[:, 2] # Gripper's Z should point opposite to object's Z
        gripper_y_axis = R_B_O[:, 1]  # Gripper's Y aligned with object's Y
        gripper_x_axis = np.cross(gripper_y_axis, gripper_z_axis)
        
        # Normalize axes to ensure a valid rotation matrix
        gripper_x_axis /= np.linalg.norm(gripper_x_axis)
        gripper_y_axis /= np.linalg.norm(gripper_y_axis)
        gripper_z_axis /= np.linalg.norm(gripper_z_axis)
        
        R_B_TCP = np.stack([gripper_x_axis, gripper_y_axis, gripper_z_axis], axis=1)
        
        T_B_TCP = np.identity(4)
        T_B_TCP[:3, :3] = R_B_TCP
        T_B_TCP[:3, 3] = target_position
        return T_B_TCP

    def _move_to_goal(self, pose_matrix, name):
        rospy.loginfo(f"Planning and moving to '{name}' pose...")
        joint_goal = self.robot_controller.get_closest_ik_solution(pose_matrix)
        
        if joint_goal is not None:
            rospy.loginfo(f"IK solution found for '{name}'. Creating and executing trajectory.")
            current_joints = self.robot_controller.get_current_joint_values()
            
            joint_trajectory_points = np.array([
                current_joints, 
                joint_goal
            ])
            self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)

            return True
        else:
            rospy.logerr(f"No IK solution found for '{name}' pose. Aborting.")
            return False
        



    def move_to_home_pose(self):
        rospy.loginfo("Moving to final HOME pose.")
        current_joints = self.robot_controller.get_current_joint_values()
        joint_trajectory_points = np.array([current_joints, HOME_POSE_JOINTS])
        self.robot_controller.send_joint_trajectory_action(joint_points=joint_trajectory_points)
        rospy.loginfo("Robot is at HOME pose.")

    def _load_pose_file(self, path):
        try:
            with open(path, 'r') as f:
                pose_type = f.read().strip()
            rospy.loginfo(f"Loaded pose type '{pose_type}' from '{path}'")
            return pose_type
        except Exception as e:
            rospy.logerr(f"Error loading pose file '{path}': {e}")
            return None

    def _load_object_file(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            rospy.loginfo(f"Loaded object data from '{path}'")
            return data
        except Exception as e:
            rospy.logerr(f"Error loading/parsing object file '{path}': {e}")
            return None

if __name__ == '__main__':
    try:
        executor = GraspExecutor()
        executor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("GraspExecutor node shutdown.")
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred in main: {e}", exc_info=True)