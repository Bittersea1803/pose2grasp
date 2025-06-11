#!/usr/bin/env python
import rospy
import numpy as np
from core.real_ur5_controller import UR5Controller

def main():
    rospy.init_node("ur5_simple_demo")
    controller = UR5Controller()

    rospy.loginfo("Starting UR5 simple demo...")

    # Move to home joint position
    home_joints = np.deg2rad([0, -90, 90, -90, -90, 0])  # example in radians
    rospy.loginfo("Moving to home position...")
    controller.move_to_joint_goal(home_joints)

    # Compute new pose offset in Z (10 cm forward in tool frame)
    current_pose = controller.get_current_tool_pose()
    T_G_0 = np.copy(current_pose)
    T_G_0[2, 3] += 0.10  # move +10 cm in Z direction

    rospy.loginfo("Planning Cartesian move 10 cm in Z...")
    joint_sol = controller.get_closest_ik_solution(T_G_0)

    if joint_sol is not None:
        controller.move_to_joint_goal(joint_sol)
    else:
        rospy.logwarn("No IK solution found for offset pose.")

    rospy.sleep(1.0)
    controller.shutdown()

if __name__ == "__main__":
    main()