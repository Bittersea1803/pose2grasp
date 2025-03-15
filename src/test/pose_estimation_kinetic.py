import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import copy
import torch
import os
import sys

pytorch_openpose_path = "/home/openpose_user/src/pose2grasp/src/pytorch-openpose"
sys.path.append(pytorch_openpose_path)

from src import util
from src.body import Body
from src.hand import Hand

class KinectOpenPose:
    def __init__(self):
        self.bridge = CvBridge()

        body_model_path = os.path.join(pytorch_openpose_path, "model", "body_pose_model.pth")
        hand_model_path = os.path.join(pytorch_openpose_path, "model", "hand_pose_model.pth")

        self.body_estimation = Body(body_model_path)
        self.hand_estimation = Hand(hand_model_path)

        if torch.cuda.is_available():
            rospy.loginfo(f"Torch device: {torch.cuda.get_device_name(0)}")
        else:
            rospy.loginfo("Torch device: CPU")

        self.sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)

    def image_callback(self, ros_image):
        try:
            oriImg = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        candidate, subset = self.body_estimation(oriImg)

        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        hands_list = util.handDetect(candidate, subset, oriImg)
        all_hand_peaks = []

        for x, y, w, is_left in hands_list:
            roi = oriImg[y:y+w, x:x+w, :]
            peaks = self.hand_estimation(roi)

            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)

        canvas = util.draw_handpose(canvas, all_hand_peaks)

        cv2.imshow("Kinect + PyTorch-OpenPose", canvas)
        cv2.waitKey(1)

def main():
    rospy.init_node("kinect_openpose_node", anonymous=True)

    node = KinectOpenPose()

    rospy.spin()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
