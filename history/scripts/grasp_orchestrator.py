#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Pose

# --- Hardcoded Configuration ---
IS_SIMULATION = True 
REQUIRED_STABILITY_TIME_SEC = 2.5
GRASP_TOPIC = "/pose2grasp/grasp_type"
OBJECT_TOPIC = "/object_detector/object_position"
ROBOT_BASE_FRAME = "base_link"
Z_OFFSET_APPROACH = 0.15
Z_OFFSET_RETREAT = 0.20

class GraspOrchestrator:
    def __init__(self):
        rospy.loginfo("Initializing Grasp Orchestrator (Standalone)...")
        if IS_SIMULATION:
            rospy.logwarn("########## ORCHESTRATOR IS IN SIMULATION MODE ##########")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.reset_state()
        
        rospy.Subscriber(GRASP_TOPIC, String, self.grasp_callback)
        rospy.Subscriber(OBJECT_TOPIC, PointStamped, self.object_callback)
        rospy.loginfo("Grasp Orchestrator is running and waiting for data.")

    def reset_state(self):
        self.last_detected_grasp = None
        self.grasp_detection_start_time = None
        self.stable_grasp_type = None
        self.last_known_object_position_cam = None
        self.is_robot_busy = False

    def object_callback(self, msg):
        if not self.is_robot_busy:
            self.last_known_object_position_cam = msg

    def grasp_callback(self, msg):
        if self.is_robot_busy: return

        detected_grasp = msg.data
        if detected_grasp != self.last_detected_grasp:
            self.last_detected_grasp = detected_grasp
            self.grasp_detection_start_time = rospy.Time.now()
            self.stable_grasp_type = None
        elif (rospy.Time.now() - self.grasp_detection_start_time).to_sec() > REQUIRED_STABILITY_TIME_SEC:
            if self.stable_grasp_type != detected_grasp:
                self.stable_grasp_type = detected_grasp
                self.check_and_trigger_robot()

    def check_and_trigger_robot(self):
        if self.stable_grasp_type and self.last_known_object_position_cam:
            rospy.loginfo("="*30)
            rospy.loginfo(f"TRIGGER! Conditions met. Starting grasp sequence.")
            rospy.loginfo(f"  -> Grasp: {self.stable_grasp_type}")
            rospy.loginfo(f"  -> Position (Camera Frame): {self.last_known_object_position_cam.point.x:.3f}, ...")
            rospy.loginfo("="*30)
            
            self.is_robot_busy = True
            self.execute_full_grasp_sequence()

    def transform_point(self, point_stamped_in):
        try:
            return self.tf_buffer.transform(point_stamped_in, ROBOT_BASE_FRAME, rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(f"TF2 transform failed: {e}")
            return None

    def execute_full_grasp_sequence(self):
        target_point_robot_frame = self.transform_point(self.last_known_object_position_cam)
        if not target_point_robot_frame:
            self.reset_state()
            return
            
        rospy.loginfo("--- STARTING SIMULATED ROBOT MOVEMENT ---")
        rospy.loginfo(f"SIM: Moving to PRE-GRASP...")
        rospy.sleep(2.0)
        rospy.loginfo("SIM: TODO: Opening gripper...")
        rospy.sleep(1.0)
        rospy.loginfo(f"SIM: Moving to GRASP...")
        rospy.sleep(1.5)
        rospy.loginfo(f"SIM: TODO: Closing gripper with '{self.stable_grasp_type}' style...")
        rospy.sleep(2.0)
        rospy.loginfo(f"SIM: Moving to RETREAT...")
        rospy.sleep(2.0)
        rospy.loginfo("--- SIMULATED MOVEMENT FINISHED ---")
        
        rospy.loginfo("SUCCESS: Grasp sequence completed!")
        self.reset_state()

if __name__ == '__main__':
    rospy.init_node('grasp_orchestrator')
    try:
        orchestrator = GraspOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
