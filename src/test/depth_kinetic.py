import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class DepthImageReceiver:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_topic = "/camera/depth/image_raw"
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)

    def depth_callback(self, msg):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge greška: %s", e)
            return

        cv_depth_normalized = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_depth_8u = cv_depth_normalized.astype('uint8')

        cv2.imshow("Kinect Depth Image", cv_depth_8u)

        colored_depth = cv2.applyColorMap(cv_depth_8u, cv2.COLORMAP_JET)
        cv2.imshow("Kinect Depth Image - ColorMap", colored_depth)

        cv2.waitKey(1)

def main():
    rospy.init_node('kinect_depth_subscriber', anonymous=True)
    receiver = DepthImageReceiver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
