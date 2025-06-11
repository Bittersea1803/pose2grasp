#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs # Potrebno za transformaciju poze
import numpy as np
# Izbrisan je quaternion_from_euler jer ga ne koristimo direktno
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped # <-- VAŽNA PROMJENA: importamo PoseStamped

# --- Hardcoded Configuration ---
IS_SIMULATION = True 
REQUIRED_STABILITY_TIME_SEC = 2.5
GRASP_TOPIC = "/pose2grasp/grasp_type"
# VAZNO: Ispravljen topic da odgovara onome sto plane_object_detector objavljuje
OBJECT_TOPIC = "/plane_object_detector/biggest_object_position"
ROBOT_BASE_FRAME = "base_link"
Z_OFFSET_APPROACH = 0.15
Z_OFFSET_RETREAT = 0.20

class GraspOrchestrator:
    def __init__(self):
        rospy.loginfo("Initializing Grasp Orchestrator...")
        if IS_SIMULATION:
            rospy.logwarn("########## ORCHESTRATOR IS IN SIMULATION MODE ##########")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.reset_state()
        
        rospy.Subscriber(GRASP_TOPIC, String, self.grasp_callback)
        # VAZNO: Promijenjen tip poruke koju subscriber očekuje
        rospy.Subscriber(OBJECT_TOPIC, PoseStamped, self.object_callback) 
        rospy.loginfo("Grasp Orchestrator is running and waiting for data.")

    def reset_state(self):
        self.last_detected_grasp = None
        self.grasp_detection_start_time = None
        self.stable_grasp_type = None
        self.last_known_object_pose_cam = None # Preimenovano radi jasnoće
        self.is_robot_busy = False

    def object_callback(self, msg: PoseStamped): # Specificiramo tip poruke
        if not self.is_robot_busy:
            self.last_known_object_pose_cam = msg

    def grasp_callback(self, msg: String):
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
        if self.stable_grasp_type and self.last_known_object_pose_cam:
            rospy.loginfo("="*30)
            rospy.loginfo(f"TRIGGER! Conditions met. Starting grasp sequence.")
            rospy.loginfo(f"  -> Grasp: {self.stable_grasp_type}")
            rospy.loginfo(f"  -> Pose (Camera Frame): {self.last_known_object_pose_cam.pose.position.x:.3f}, ...")
            rospy.loginfo("="*30)
            
            self.is_robot_busy = True
            self.execute_full_grasp_sequence()

    def transform_pose(self, pose_stamped_in: PoseStamped): # Funkcija sada transformira cijelu pozu
        try:
            # tf2 automatski zna kako transformirati PoseStamped poruku
            return self.tf_buffer.transform(pose_stamped_in, ROBOT_BASE_FRAME, rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(f"TF2 transform failed: {e}")
            return None

    def execute_full_grasp_sequence(self):
        # Transformiramo cijelu pozu, ne samo točku
        target_pose_robot_frame = self.transform_pose(self.last_known_object_pose_cam)
        if not target_pose_robot_frame:
            rospy.logerr("Could not transform object pose to robot frame. Aborting.")
            self.reset_state()
            return
            
        rospy.loginfo("--- STARTING SIMULATED ROBOT MOVEMENT ---")
        rospy.loginfo(f"SIM: Target pose in '{ROBOT_BASE_FRAME}': Position=[{target_pose_robot_frame.pose.position.x:.2f}, {target_pose_robot_frame.pose.position.y:.2f}, {target_pose_robot_frame.pose.position.z:.2f}], Orientation=[{target_pose_robot_frame.pose.orientation.w:.2f}, ...]")
        
        rospy.loginfo(f"SIM: Moving to PRE-GRASP...")
        rospy.sleep(2.0)
        # ... ostatak simulacije ostaje isti ...
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