import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class KinectImageReceiver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/camera/rgb/image_color"
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        cv2.imshow("Kinect", cv_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('kinect_image_subscriber', anonymous=True)
    receiver = KinectImageReceiver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()