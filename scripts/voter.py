import os
import sys
import time
import threading
from typing import Optional, Tuple, Dict, Any, List 

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import joblib 

# --- Postavke za OpenPose ---
OPENPOSE_REPO = "/home/openpose_user/src/pose2grasp/src/pytorch-openpose"
try:
    if not os.path.isdir(OPENPOSE_REPO):
        alt_openpose_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "pytorch-openpose")
        if os.path.isdir(alt_openpose_repo): OPENPOSE_REPO = alt_openpose_repo
        else: raise ImportError(f"OpenPose repo directory not found at: {OPENPOSE_REPO} or {alt_openpose_repo}")
    if OPENPOSE_REPO not in sys.path: sys.path.append(OPENPOSE_REPO)
    openpose_src_dir = os.path.join(OPENPOSE_REPO, "src")
    if os.path.exists(os.path.join(openpose_src_dir, "__init__.py")):
        if openpose_src_dir not in sys.path: sys.path.insert(0, openpose_src_dir) 
        from hand import Hand
        rospy.loginfo("OpenPose importiran kao paket (iz src).")
    else:
        if os.path.isdir(openpose_src_dir) and openpose_src_dir not in sys.path:
             sys.path.insert(0, openpose_src_dir)
        from hand import Hand 
        rospy.logwarn("Importing OpenPose module 'hand'.")
except ImportError as e:
    print(f"Greška: Nije moguće importati OpenPose iz {OPENPOSE_REPO}. Detalji: {e}")
    sys.exit(1)
except FileNotFoundError as e: 
    print(f"Greška: OpenPose repozitorij ili datoteke nisu pronađene. {e}")
    sys.exit(1)

# --- Putanje do spremljenih modela ---
BASE_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODEL_DIR = os.path.join(BASE_PROJECT_DIR, "models")
MODEL_FILENAME = "xgboost_best_model.joblib" 
IMPUTER_FILENAME = "imputer.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
IMPUTER_PATH = os.path.join(MODEL_DIR, IMPUTER_FILENAME)
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, LABEL_ENCODER_FILENAME)

# --- ROS i konstante ---
RGB_TOPIC_RECT_COLOR = "/camera/rgb/image_rect_color"
REGISTERED_DEPTH_TOPIC = "/camera/depth_registered/hw_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info" 
MESSAGE_FILTER_SLOP = 0.2 
VALID_DEPTH_THRESHOLD_MM = (400, 1500) 
DEPTH_NEIGHBORHOOD_SIZE = 3 
DEPTH_STD_DEV_THRESHOLD_MM = 35.0
OPENPOSE_CONFIDENCE_THRESHOLD = 0.2

class ModelTester:
    def __init__(self):
        rospy.init_node("model_tester_node", anonymous=True)
        self.bridge = CvBridge()
        rospy.loginfo("ROS čvor za testiranje modela inicijaliziran.")
        self.is_shutting_down = False
        
        self._load_artifacts() 
        self._initialize_openpose_estimator()

        self._latest_synced_data: Dict[str, Any] = {"rgb": None, "depth": None}
        self._data_lock = threading.Lock()
        self._fx: Optional[float] = None; self._fy: Optional[float] = None
        self._cx: Optional[float] = None; self._cy: Optional[float] = None
        self._camera_info_received: bool = False
        
        self.last_printed_pose_time = time.time()
        self.print_interval = 1.0 # Ispisuj predikciju svake sekunde

        self._initialize_ros_subscriptions()
        
        # Nema GUI-ja, pa nema ni GUI petlje. Glavna obrada će biti u ROS callbacku ili zasebnoj dretvi.
        # Za jednostavnost, obradu ćemo staviti u callback, ali ćemo paziti da ne blokira predugo.
        # Ili, još bolje, imat ćemo zasebnu dretvu za obradu.

        self.processing_thread = threading.Thread(target=self._continuous_processing, daemon=True)
        self.processing_thread.start()

        rospy.on_shutdown(self.on_close)


    def _load_artifacts(self):
        try:
            if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model nije pronađen: {MODEL_PATH}")
            if not os.path.exists(IMPUTER_PATH): raise FileNotFoundError(f"Imputer nije pronađen: {IMPUTER_PATH}")
            if not os.path.exists(LABEL_ENCODER_PATH): raise FileNotFoundError(f"Label encoder nije pronađen: {LABEL_ENCODER_PATH}")

            self.model = joblib.load(MODEL_PATH)
            self.imputer = joblib.load(IMPUTER_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            rospy.loginfo("Model, imputer i label encoder uspješno učitani.")
        except FileNotFoundError as e:
            msg = f"Nije pronađena datoteka: {e}. Jeste li pokrenuli skriptu za spremanje (`train_and_save_models.py`)?"
            rospy.logfatal(msg)
            sys.exit(1) 
        except Exception as e:
            msg = f"Došlo je do greške prilikom učitavanja artefakata: {e}"
            rospy.logfatal(msg)
            sys.exit(1)

    def _initialize_openpose_estimator(self):
        try:
            openpose_model_path = os.path.join(OPENPOSE_REPO, "model", "hand_pose_model.pth")
            if not os.path.exists(openpose_model_path):
                raise FileNotFoundError(f"OpenPose model datoteka nije pronađena na: {openpose_model_path}")
            self._hand_estimator = Hand(openpose_model_path)
            rospy.loginfo("OpenPose estimator uspješno inicijaliziran.")
        except Exception as e:
            msg = f"Neuspjela inicijalizacija OpenPose-a: {e}"
            rospy.logfatal(msg)
            sys.exit(1)

    def _initialize_ros_subscriptions(self):
        self.rgb_sub_filter = message_filters.Subscriber(RGB_TOPIC_RECT_COLOR, Image)
        self.depth_sub_filter = message_filters.Subscriber(REGISTERED_DEPTH_TOPIC, Image)
        self.info_sub_filter = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_filter, self.depth_sub_filter, self.info_sub_filter],
            queue_size=5, slop=MESSAGE_FILTER_SLOP 
        )
        self.ts.registerCallback(self._synchronized_ros_callback)
        rospy.loginfo("ROS pretplate inicijalizirane.")

    def _synchronized_ros_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self.is_shutting_down: return
        
        if not self._camera_info_received and hasattr(info_msg, 'K') and len(info_msg.K) == 9:
            self._fx, self._fy = info_msg.K[0], info_msg.K[4]
            self._cx, self._cy = info_msg.K[2], info_msg.K[5]
            if self._fx > 0 and self._fy > 0:
                self._camera_info_received = True
                rospy.loginfo(f"Primljeni parametri kamere: fx={self._fx:.2f}, fy={self._fy:.2f}, cx={self._cx:.2f}, cy={self._cy:.2f}")
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            with self._data_lock:
                self._latest_synced_data["rgb"] = cv_rgb 
                self._latest_synced_data["depth"] = cv_depth
        except Exception as e:
            rospy.logerr_throttle(5, f"Greška u _synchronized_ros_callback: {e}")

    def _continuous_processing(self):
        rate = rospy.Rate(10) # Pokušaj obraditi 10 puta u sekundi (može se smanjiti ako je presporo)
        while not rospy.is_shutdown() and not self.is_shutting_down:
            current_rgb_frame: Optional[np.ndarray] = None
            current_depth_frame: Optional[np.ndarray] = None

            with self._data_lock:
                if self._latest_synced_data["rgb"] is not None:
                    current_rgb_frame = self._latest_synced_data["rgb"].copy()
                if self._latest_synced_data["depth"] is not None:
                    current_depth_frame = self._latest_synced_data["depth"].copy()

            if current_rgb_frame is not None and current_depth_frame is not None and self._camera_info_received:
                try:
                    # --- Opcionalno: Prikaz slike s cv2.imshow ---
                    # cv2.imshow("Live Stream", current_rgb_frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.on_close()
                    #     break
                    # --- Kraj opcionalnog prikaza ---

                    # Smanjenje rezolucije za OpenPose (OPCIJA ZA UBRZANJE)
                    # visina_orig, sirina_orig = current_rgb_frame.shape[:2]
                    # frame_za_openpose = cv2.resize(current_rgb_frame, (sirina_orig // 2, visina_orig // 2), interpolation=cv2.INTER_AREA)
                    # all_peaks_2d_small = self._hand_estimator(frame_za_openpose)
                    # if all_peaks_2d_small is not None and isinstance(all_peaks_2d_small, np.ndarray):
                    #     all_peaks_2d = all_peaks_2d_small.copy()
                    #     all_peaks_2d[:, 0] *= 2 
                    #     all_peaks_2d[:, 1] *= 2
                    # else:
                    #     all_peaks_2d = None
                    
                    all_peaks_2d = self._hand_estimator(current_rgb_frame) # Koristi originalnu rezoluciju

                    if all_peaks_2d is None or not isinstance(all_peaks_2d, np.ndarray) or all_peaks_2d.ndim != 2:
                        predicted_pose_label = "- (Nema detekcije ruke)"
                    else:
                        peaks_for_projection = np.full((21, 2), np.nan, dtype=np.float32)
                        num_detected_peaks = 0
                        for i in range(min(21, all_peaks_2d.shape[0])):
                            if all_peaks_2d.shape[1] < 3 or all_peaks_2d[i, 2] >= OPENPOSE_CONFIDENCE_THRESHOLD:
                                peaks_for_projection[i, :] = all_peaks_2d[i, :2]
                                num_detected_peaks +=1
                        
                        if num_detected_peaks < 10: 
                            predicted_pose_label = f"- (Premalo točaka: {num_detected_peaks})"
                        else:        
                            keypoints_3d_cam = np.full((21, 3), np.nan, dtype=np.float32)
                            valid_3d_points_count = 0
                            for i in range(21):
                                if not np.isnan(peaks_for_projection[i, 0]):
                                    u, v = peaks_for_projection[i, 0], peaks_for_projection[i, 1]
                                    z_m = self._get_depth_from_neighborhood(current_depth_frame, u, v)
                                    if not np.isnan(z_m) and self._fx and self._fy and self._cx and self._cy: 
                                        x_cam = (u - self._cx) * z_m / self._fx
                                        y_cam = (v - self._cy) * z_m / self._fy
                                        keypoints_3d_cam[i] = [x_cam, y_cam, z_m]
                                        valid_3d_points_count +=1
                            
                            if valid_3d_points_count < 5: 
                                predicted_pose_label = f"- (Premalo 3D točaka: {valid_3d_points_count})"
                            elif np.isnan(keypoints_3d_cam[0]).any():
                                predicted_pose_label = "- (Nema 3D zgloba)"
                            else:
                                wrist_3d = keypoints_3d_cam[0].copy()
                                keypoints_rel_3d = keypoints_3d_cam - wrist_3d
                                keypoints_rel_3d[0] = [0.0, 0.0, 0.0] 

                                features_flat = keypoints_rel_3d.flatten()
                                features_imputed_flat = self.imputer.transform(features_flat.reshape(1, -1)).flatten()
                                prediction_encoded = self.model.predict(features_imputed_flat.reshape(1,-1))
                                predicted_pose_label = self.label_encoder.inverse_transform(prediction_encoded)[0].upper()
                    
                    current_time = time.time()
                    if current_time - self.last_printed_pose_time >= self.print_interval:
                        rospy.loginfo(f"Predviđena poza: {predicted_pose_label}")
                        self.last_printed_pose_time = current_time

                except Exception as e:
                    rospy.logerr(f"Greška u _continuous_processing (klasifikacija): {e}")
                    import traceback; rospy.logerr(traceback.format_exc())
            
            rate.sleep() # Održava frekvenciju petlje


    def _get_depth_from_neighborhood(self, depth_map: np.ndarray, cx: float, cy: float, size: int = DEPTH_NEIGHBORHOOD_SIZE) -> float:
        if depth_map is None: return np.nan
        if size % 2 == 0: size += 1
        radius = size // 2
        h, w = depth_map.shape
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < w and 0 <= iy < h): return np.nan
        y_min, y_max = max(0, iy - radius), min(h, iy + radius + 1)
        x_min, x_max = max(0, ix - radius), min(w, ix + radius + 1)
        neighborhood = depth_map[y_min:y_max, x_min:x_max]
        min_d_thresh, max_d_thresh = VALID_DEPTH_THRESHOLD_MM 
        valid_depths_mm = neighborhood[(neighborhood >= min_d_thresh) & (neighborhood <= max_d_thresh)]
        if valid_depths_mm.size < max(1, (size*size)//4): return np.nan
        std_dev_mm = np.std(valid_depths_mm)
        if std_dev_mm > DEPTH_STD_DEV_THRESHOLD_MM: 
            return np.nan
        return float(np.median(valid_depths_mm) / 1000.0)

    # Nema više _create_overlay_image, _resize_for_display, _update_gui_main_preview, _pump_gui_queue_periodically
    # ako ne koristimo Tkinter GUI za prikaz slike.

    def on_close(self): # Poziva se kada ROS čvor primi signal za gašenje
        if not self.is_shutting_down:
            self.is_shutting_down = True
            rospy.loginfo("Zatvaranje jednostavnog klasifikatora poza...")
            if hasattr(self, 'ts') and self.ts is not None:
                try:
                    if hasattr(self, 'rgb_sub_filter') and self.rgb_sub_filter: self.rgb_sub_filter.sub.unregister()
                    if hasattr(self, 'depth_sub_filter') and self.depth_sub_filter: self.depth_sub_filter.sub.unregister()
                    if hasattr(self, 'info_sub_filter') and self.info_sub_filter: self.info_sub_filter.sub.unregister()
                except Exception as e: rospy.logwarn(f"Greška pri odjavi subscribera: {e}")
                self.ts = None
                rospy.loginfo("ROS pretplatnici odjavljeni.")
            
            # Nema GUI prozora za uništavanje u ovoj verziji
            rospy.loginfo("Aplikacija klasifikatora ugašena.")
            # Ne treba zvati rospy.signal_shutdown() ovdje jer se ovo poziva iz on_shutdown hook-a


if __name__ == "__main__":
    tester_app = None
    try:
        tester_app = ModelTester()
        rospy.spin() # Umjesto tester_app.run(), sada koristimo rospy.spin() jer nemamo Tkinter mainloop
    except rospy.ROSInterruptException: rospy.loginfo("ROS čvor testera prekinut.")
    except KeyboardInterrupt: rospy.loginfo("Tester prekinut od strane korisnika (Ctrl+C).")
    except SystemExit: rospy.loginfo("Izlazak iz aplikacije zbog greške pri inicijalizaciji.")
    except Exception as e:
        rospy.logfatal(f"Neočekivana greška u __main__ testera: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rospy.loginfo("__main__ finally blok testera dosegnut.")
        if tester_app and hasattr(tester_app, 'is_shutting_down') and not tester_app.is_shutting_down:
            # on_close će biti pozvan od strane rospy.on_shutdown hook-a
            pass 
        
        if not rospy.is_shutdown(): # Dodatna provjera
            rospy.signal_shutdown("Kraj __main__ bloka testera.")
        rospy.loginfo(f"Skripta {os.path.basename(__file__)} potpuno završena.")