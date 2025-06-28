import rospy
import numpy as np
import moveit_commander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_srvs.srv import Trigger
from core.transforms import pose_to_matrix, matrix_to_pose
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading
import RVLPYDDManipulator as rvlpy_dd_man
import roslib

roslib.load_manifest('robotiq_3f_gripper_control')
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotOutput

class UR5Controller:
    def __init__(self, move_group="arm", 
                 action_topic="/scaled_pos_joint_traj_controller/follow_joint_trajectory",
                 rvl_cfg_path="/home/RVLuser/rvl-linux/RVLMotionDemo_Cupec.cfg"):
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(move_group)

        # RVL IK solver
        self.rvl_ik_solver = rvlpy_dd_man.PYDDManipulator()
        self.rvl_ik_solver.create(rvl_cfg_path)
        # self.rvl_ik_solver.set_robot_pose(T_R_W)

        self.joint_names = self.group.get_active_joints()
        self.joint_states = None
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        self.client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for trajectory action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to FollowJointTrajectory action server.")

        self.active_goal = None
        self.latest_wrench = None
        self.force_threshold = 30.0  # Newtons
        self.force_violation = False
        # Use real FT sensor topic
        rospy.Subscriber("/robotiq_ft_sensor/wrench", WrenchStamped, self.wrench_callback)

        self.gripper_pub = rospy.Publisher('Robotiq3FGripperRobotOutput', Robotiq3FGripperRobotOutput, queue_size=10)
        rospy.loginfo("Robotiq gripper publisher initialized.")

        self.add_ground_plane()
        
    def activate_gripper(self):
        rospy.loginfo("Activating Robotiq Gripper...")
        cmd = Robotiq3FGripperRobotOutput()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rMOD = 0
        cmd.rPRA = 0
        cmd.rSPA = 255
        cmd.rFRA = 150
        cmd.rICF = cmd.rICS = cmd.rATR = 0
        cmd.rPRB = cmd.rPRC = 0
        cmd.rSPB = cmd.rSPC = 0
        cmd.rFRB = cmd.rFRC = 0
        cmd.rPRS = cmd.rSPS = cmd.rFRS = 0
        while self.gripper_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("AG: Waiting for gripper subscriber connection...")
            rospy.sleep(0.1)
        self.gripper_pub.publish(cmd)
        rospy.sleep(2.0)
        rospy.loginfo("Gripper activated.")

    def send_gripper_command(self, grasp_type, position, speed=255, force=150):
        cmd = Robotiq3FGripperRobotOutput()
        mode_map = {'basic': 0, 'pinch': 1, 'wide': 2, 'scissor': 3}
        mode = mode_map.get(grasp_type.lower(), 0)
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rMOD = mode
        cmd.rPRA = int(np.clip(position, 0, 255))
        cmd.rSPA = int(np.clip(speed, 0, 255))
        cmd.rFRA = int(np.clip(force, 0, 255))
        cmd.rICF = cmd.rICS = cmd.rATR = 0
        cmd.rPRB = cmd.rPRC = 0
        cmd.rSPB = cmd.rSPC = 0
        cmd.rFRB = cmd.rFRC = 0
        cmd.rPRS = cmd.rSPS = cmd.rFRS = 0
        while self.gripper_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("SGC: Waiting for gripper subscriber connection...")
            rospy.sleep(0.1)
        self.gripper_pub.publish(cmd)
        rospy.sleep(1.5)


    def add_ground_plane(self, z_height=0.0, size=(2.0, 2.0), timeout=2.0):
        """
        Adds a ground plane box to the planning scene at the specified height.
        """
        rospy.loginfo("Adding ground plane to planning scene...")

        ground_pose = PoseStamped()
        ground_pose.header.frame_id = self.robot.get_planning_frame()  # usually "base_link"
        ground_pose.pose.orientation.w = 1.0
        ground_pose.pose.position.x = 0.0
        ground_pose.pose.position.y = 0.0
        ground_pose.pose.position.z = z_height - 0.01  # slightly below to act as "floor"

        self.scene.add_box("ground_plane", ground_pose, size=(size[0], size[1], 0.02))

        # Wait for scene update
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and (rospy.get_time() - start_time < timeout):
            if "ground_plane" in self.scene.get_known_object_names():
                rospy.loginfo("Ground plane added.")
                return True
            rospy.sleep(0.1)

        rospy.logwarn("Timed out waiting for ground plane to be added.")
        return False

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def wrench_callback(self, msg):
        self.latest_wrench = msg.wrench

    def get_current_joint_values(self):
        return np.array(self.group.get_current_joint_values())

    def get_current_tool_pose(self):
        pose_msg = self.group.get_current_pose().pose
        return pose_to_matrix(pose_msg)

    def move_to_joint_goal(self, joint_goal: np.ndarray):
        assert joint_goal.shape == (6,)
        self.group.go(joint_goal.tolist(), wait=True)
        self.group.stop()

    def move_to_pose_matrix(self, pose_matrix: np.ndarray):
        assert pose_matrix.shape == (4, 4)
        pose_msg = matrix_to_pose(pose_matrix)
        self.group.set_pose_target(pose_msg)
        self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

    def send_joint_trajectory_action(self, joint_points: np.ndarray, max_velocity=1.0, max_acceleration=1.2) -> bool:
        assert joint_points.shape[1] == 6

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        cumulative_time = 0.0

        for i, point in enumerate(joint_points):
            pt = JointTrajectoryPoint()
            pt.positions = point.tolist()
            dt = self.calculate_time_between_points(joint_points[i - 1], point, max_velocity, max_acceleration) if i > 0 else 0.0
            cumulative_time += dt
            pt.time_from_start = rospy.Duration(cumulative_time)
            traj.points.append(pt)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.force_violation = False

        try:
            rospy.loginfo("Sending trajectory goal...")
            self.client.send_goal(goal)
            self.active_goal = self.client

            self.client.wait_for_result()

            result = self.client.get_result()
            state = self.client.get_state()

            if self.force_violation:
                rospy.logwarn("Trajectory canceled due to force threshold.")
                return False

            if state == 3:  # SUCCEEDED
                rospy.loginfo("Trajectory executed successfully.")
                return True
            else:
                rospy.logwarn("Trajectory failed or was interrupted. State: %s", str(state))
                return False
        except Exception as e:
            rospy.logerr("Trajectory execution error: %s", str(e))
            return False
    
    def cancel_trajectory(self):
        if self.active_goal:
            rospy.logwarn("Cancelling active trajectory...")
            self.active_goal.cancel_goal()

    def zero_ft_sensor(self) -> bool:
        """
        Call the external FT sensor zeroing service.
        Returns True if successful, False otherwise.
        """
        service_name = "/robotiq_ft_streamer/zero"
        rospy.loginfo("[UR5Controller] Calling FT sensor zeroing service...")

        try:
            rospy.wait_for_service(service_name, timeout=3.0)
            zero_service = rospy.ServiceProxy(service_name, Trigger)
            response = zero_service()
            if response.success:
                rospy.loginfo("[UR5Controller] FT sensor zeroed: %s", response.message)
                return True
            else:
                rospy.logwarn("[UR5Controller] FT zeroing failed: %s", response.message)
                return False
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("[UR5Controller] Failed to call FT zeroing service: %s", str(e))
            return False

    def get_current_wrench(self) -> np.ndarray:
        """
        Returns the latest force-torque values as a 6-element numpy array:
        [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        if self.latest_wrench is None:
            rospy.logwarn_throttle(5, "[UR5Controller] No FT data received yet.")
            return np.zeros(6)

        f = self.latest_wrench.force
        t = self.latest_wrench.torque
        return np.array([f.x, f.y, f.z, t.x, t.y, t.z])

    def get_all_ik_solutions(self, pose_matrix: np.ndarray):
        """
        Compute all inverse kinematics solutions using RVL IK solver.
        
        Args:
            pose_matrix (np.ndarray): 4x4 transformation matrix representing desired end-effector pose (T_G_0)

        Returns:
            solutions (np.ndarray): Nx6 array of joint solutions
            success (bool): whether IK computation was successful
        """
        assert pose_matrix.shape == (4, 4), "Expected 4x4 transformation matrix"

        try:
            q_solutions, num_solutions, success = self.rvl_ik_solver.inv_kinematics_all_sols_prev(pose_matrix)

            if not success or num_solutions == 0:
                rospy.logwarn("RVL IK solver failed to find solutions.")
                return None, False

            # Reshape and trim based on number of solutions
            q_solutions = np.array(q_solutions).reshape((8, -1))[:num_solutions]
            return q_solutions, True

        except Exception as e:
            rospy.logerr("RVL IK solver error: %s", str(e))
            return None, False

    def get_closest_ik_solution(self, T_G_0: np.ndarray) -> np.ndarray:
        """
        Uses the RVL IK solver to get all solutions for the given pose,
        and returns the one closest to the current joint configuration
        using Chebyshev (L∞) distance.

        Args:
            T_G_0 (np.ndarray): 4x4 pose matrix of the end-effector in base frame

        Returns:
            np.ndarray: 6-element joint solution closest to current config
        """
        if T_G_0.shape != (4, 4):
            raise ValueError("Expected a 4x4 transformation matrix for T_G_0")

        # Get all IK solutions using RVL solver
        q_solutions, num_sols, success = self.rvl_ik_solver.inv_kinematics_all_sols_prev(T_G_0)

        if not success or num_sols == 0:
            rospy.logwarn("RVL IK solver found no solutions.")
            return None

        q_solutions = np.array(q_solutions[:num_sols, ...])
        q_solutions[:, 0] -= np.pi
        # q_solutions[:, l5] -= np.pi
        q_solutions[q_solutions > np.pi] -= (2.0*np.pi)
        q_solutions[q_solutions < -np.pi] += (2.0*np.pi)
        np.unwrap(q_solutions, axis=0)
        
        # Get current joint configuration
        got_joints = False
        for i in range(100):
            q_current = self.get_current_joint_values()
            if q_current.shape[0] > 0:
                got_joints = True
                break
        if not got_joints:
            raise RuntimeError("Failed to get current joint values after 100 attempts.")
        
        # Compute Chebyshev (L∞) distance for each solution
        cheb_dists = np.max(np.abs(q_solutions - q_current), axis=1)
        closest_idx = np.argmin(cheb_dists)

        closest_q = q_solutions[closest_idx]

        rospy.loginfo("Found %d IK solutions. Closest (Chebyshev dist: %.4f)", num_sols, cheb_dists[closest_idx])

        return closest_q

    def shutdown(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("UR5Controller shutdown complete.")

    @staticmethod
    def calculate_time_between_points(start, end, max_velocity, max_acceleration):
        position_differences = np.abs(np.array(end) - np.array(start))
        time_velocity = position_differences / max_velocity
        time_acceleration = np.sqrt(2 * position_differences / max_acceleration)
        return max(np.max(time_velocity), np.max(time_acceleration))