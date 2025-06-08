#!/usr/bin/env python
import rospy
import joblib
import numpy as np
import pandas as pd
from collections import deque
from std_msgs.msg import String
from ros_openpose.msg import Frame

class Voter:
    """
    A simple class to find the most frequent element in a buffer.
    This is used to 'smooth' the output of the grasp classifier.
    """
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)

    def add_vote(self, vote):
        """Adds a new vote to the buffer."""
        self.buffer.append(vote)

    def get_dominant_vote(self):
        """Returns the most frequent element in the buffer."""
        if not self.buffer:
            return None
        return max(set(self.buffer), key=self.buffer.count)

class LiveGraspDetector:
    """
    This ROS node subscribes to OpenPose hand keypoints, classifies the grasp
    type using a pre-trained model, and publishes the result.
    """
    def __init__(self):
        rospy.loginfo("Initializing Live Grasp Detector...")
        
        # Load parameters from ROS Parameter Server
        self.load_params()
        
        # Load the trained model
        self.model = self.load_model()
        
        # Initialize the Voter for smoothing predictions
        self.voter = Voter(buffer_size=self.voter_buffer_size)
        
        # ROS Publisher and Subscriber
        self.grasp_publisher = rospy.Publisher(self.output_topic, String, queue_size=10)
        rospy.Subscriber(self.openpose_topic, Frame, self.frame_callback)

        rospy.loginfo("Live Grasp Detector is running.")

    def load_params(self):
        """Loads parameters with default values."""
        self.model_path = rospy.get_param('~model_path', 'models/xgboost/xgboost_model.joblib')
        self.openpose_topic = rospy.get_param('~openpose_topic', '/ros_openpose/frame')
        self.output_topic = rospy.get_param('~output_topic', '/pose2grasp/grasp_type')
        self.voter_buffer_size = rospy.get_param('~voter_buffer_size', 15)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.75)
        rospy.loginfo(f"Confidence threshold set to: {self.confidence_threshold}")

    def load_model(self):
        """Loads the pre-trained classifier model from disk."""
        try:
            # Note: The model path is relative to the catkin_workspace root
            # You might need to provide an absolute path or use package-relative paths
            # For now, we assume it's run from the workspace root.
            # A better way is to use rospkg to find the package path.
            import rospkg
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('pose2grasp')
            absolute_model_path = f"{package_path}/{self.model_path}"
            
            model = joblib.load(absolute_model_path)
            rospy.loginfo(f"Model successfully loaded from: {absolute_model_path}")
            return model
        except Exception as e:
            rospy.logerr(f"Failed to load model from {self.model_path}: {e}")
            rospy.signal_shutdown("Model not found")
            return None

    def frame_callback(self, data):
        """Processes each frame from OpenPose."""
        if not self.model:
            return

        # Ensure a person and a hand are detected
        if not data.persons or not data.persons[0].hand_key_points_2d:
            return

        try:
            # Flatten the keypoints into a single list [x1, y1, x2, y2, ...]
            key_points = data.persons[0].hand_key_points_2d
            coords = [c for point in key_points for c in (point.x, point.y)]
            
            df = pd.DataFrame([coords])
            
            # Use predict_proba to get confidence scores
            probabilities = self.model.predict_proba(df)[0]
            max_probability = np.max(probabilities)
            
            # Only consider predictions above the confidence threshold
            if max_probability >= self.confidence_threshold:
                prediction_index = np.argmax(probabilities)
                predicted_class = self.model.classes_[prediction_index]
                self.voter.add_vote(predicted_class)
            
            # Publish the smoothed, dominant grasp type
            dominant_grasp = self.voter.get_dominant_vote()
            if dominant_grasp:
                self.grasp_publisher.publish(dominant_grasp)

        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")

if __name__ == '__main__':
    rospy.init_node('live_grasp_detector')
    try:
        LiveGraspDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass