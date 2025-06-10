import rospy
import joblib
import numpy as np
import pandas as pd
from collections import deque
import message_filters
from cv_bridge import CvBridge

# ROS Poruke
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from ros_openpose.msg import Frame # Pretpostavka da OpenPose objavljuje na ovu temu


try:
    from src.hand import Hand
except ImportError:
    rospy.logerr("Could not import OpenPose 'Hand' estimator. Make sure it's in your PYTHONPATH.")
    Hand = None

# --- Hardcoded Configuration ---
# Ove vrijednosti se mogu prebaciti u launch file kao parametri kasnije
MODEL_PATH = "../models/xgboost/xgboost_model.joblib"
# Teme za sinkronizaciju
RGB_TOPIC = "/camera/rgb/image_rect_color"
DEPTH_TOPIC = "/camera/depth_registered/image_rect_raw"
INFO_TOPIC = "/camera/rgb/camera_info"
OPENPOSE_TOPIC = "/ros_openpose/frame" # Tema na koju OpenPose objavljuje 2D točke
OUTPUT_TOPIC = "/pose2grasp/grasp_type"

VOTER_BUFFER_SIZE = 15
CONFIDENCE_THRESHOLD = 0.75
# Pragovi za dubinu, kao u data_collect.py
VALID_DEPTH_THRESHOLD_MM = (400, 1500)
DEPTH_NEIGHBORHOOD_SIZE = 3

class Voter:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
    def add_vote(self, vote):
        self.buffer.append(vote)
    def get_dominant_vote(self):
        if not self.buffer: return None
        return max(set(self.buffer), key=self.buffer.count)

class LiveGraspDetector3D:
    def __init__(self):
        rospy.loginfo("Initializing 3D Live Grasp Detector...")
        if Hand is None:
            rospy.signal_shutdown("OpenPose not available.")
            return

        self.bridge = CvBridge()
        self.model = self.load_model()
        self.hand_estimator = Hand("../src/pytorch-openpose/model/hand_pose_model.pth") # Prilagodi putanju
        self.voter = Voter(buffer_size=VOTER_BUFFER_SIZE)

        # Kamera parametri
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.camera_info_received = False
        rospy.Subscriber(INFO_TOPIC, CameraInfo, self.info_callback)
        rospy.loginfo("Waiting for camera info...")
        
        # Sinkronizacija RGB i Depth slike
        self.rgb_sub = message_filters.Subscriber(RGB_TOPIC, Image)
        self.depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1
        )
        self.ts.registerCallback(self.image_callback)
        
        self.grasp_publisher = rospy.Publisher(OUTPUT_TOPIC, String, queue_size=10)
        rospy.loginfo("3D Live Grasp Detector is running.")

    def info_callback(self, msg):
        """Jednokratni callback za spremanje intrinzičnih parametara kamere."""
        if not self.camera_info_received:
            self.fx, self.fy = msg.K[0], msg.K[4]
            self.cx, self.cy = msg.K[2], msg.K[5]
            self.camera_info_received = True
            rospy.loginfo(f"Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def load_model(self):
        # ... (ista load_model funkcija kao prije)
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            rospy.loginfo(f"Model successfully loaded from: {MODEL_PATH}")
            return model
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            rospy.signal_shutdown("Model not found")
            return None

    def image_callback(self, rgb_msg, depth_msg):
        """Callback koji se poziva kada stignu sinkronizirane slike."""
        if not self.camera_info_received or not self.model:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        # 1. Pronađi 2D točke ruke pomoću OpenPose
        all_peaks_2d = self.hand_estimator(rgb_image)
        if all_peaks_2d is None:
            return

        # 2. Projektiraj 2D točke u 3D prostor
        keypoints_3d_cam_frame = np.full((21, 3), np.nan, dtype=np.float32)
        for i in range(min(21, all_peaks_2d.shape[0])):
            # Provjeri pouzdanost detekcije
            if all_peaks_2d.shape[1] >= 3 and all_peaks_2d[i, 2] < 0.2: # Prag pouzdanosti
                continue

            u, v = int(round(all_peaks_2d[i, 0])), int(round(all_peaks_2d[i, 1]))
            
            # Uzmi dubinu iz susjedstva, kao u data_collect.py
            z_meters = self.get_depth_from_neighborhood(depth_image, u, v)
            if not np.isnan(z_meters):
                x_cam = (u - self.cx) * z_meters / self.fx
                y_cam = (v - self.cy) * z_meters / self.fy
                keypoints_3d_cam_frame[i] = [x_cam, y_cam, z_meters]

        # 3. Izračunaj relativne koordinate u odnosu na zglob
        if np.isnan(keypoints_3d_cam_frame[0]).any():
            return # Ako nemamo zglob, ne možemo ništa

        wrist_3d = keypoints_3d_cam_frame[0].copy()
        keypoints_rel_3d = keypoints_3d_cam_frame - wrist_3d
        keypoints_rel_3d[0] = [0.0, 0.0, 0.0]

        # 4. Pripremi podatke za model i klasificiraj
        features_flat = keypoints_rel_3d.flatten()
        if np.isnan(features_flat).any():
            # XGBoost može raditi s NaN vrijednostima, pa je ovo OK
            pass

        df = pd.DataFrame([features_flat])
        
        probabilities = self.model.predict_proba(df)[0]
        max_probability = np.max(probabilities)
        
        if max_probability >= CONFIDENCE_THRESHOLD:
            prediction_index = np.argmax(probabilities)
            predicted_class = self.model.classes_[prediction_index]
            self.voter.add_vote(predicted_class)
        
        dominant_grasp = self.voter.get_dominant_vote()
        if dominant_grasp:
            self.grasp_publisher.publish(dominant_grasp)

    def get_depth_from_neighborhood(self, depth_map_mm, u_px, v_px):
        # ... (ista get_depth_from_neighborhood funkcija kao u data_collect.py)
        if not (0 <= v_px < depth_map_mm.shape[0] and 0 <= u_px < depth_map_mm.shape[1]):
            return np.nan
        
        radius = DEPTH_NEIGHBORHOOD_SIZE // 2
        y_start, y_end = max(0, v_px - radius), min(depth_map_mm.shape[0], v_px + radius + 1)
        x_start, x_end = max(0, u_px - radius), min(depth_map_mm.shape[1], u_px + radius + 1)
        
        neighborhood = depth_map_mm[y_start:y_end, x_start:x_end]
        valid_depths = neighborhood[(neighborhood >= VALID_DEPTH_THRESHOLD_MM[0]) & (neighborhood <= VALID_DEPTH_THRESHOLD_MM[1])]
        
        if valid_depths.size < 3: return np.nan
        
        return float(np.median(valid_depths) / 1000.0)

if __name__ == '__main__':
    rospy.init_node('live_grasp_detector_3d')
    try:
        LiveGraspDetector3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass