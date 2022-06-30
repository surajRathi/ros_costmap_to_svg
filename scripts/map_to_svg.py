#! /usr/bin/python3
import cv2
import rospy
from rospy.numpy_msg import numpy_msg

import numpy as np
import cv2 as cv
from nav_msgs.msg import OccupancyGrid


def callback(msg: OccupancyGrid):
    rospy.loginfo("Received a cost map")
    map_info = msg.info

    img = msg.data.reshape(map_info.height, map_info.width, 1).astype(np.uint8)
    cv2.imshow("msg", img)
    cv2.waitKey()

    rospy.signal_shutdown("work done")


# def process_image()
def main():
    rospy.init_node("map_to_svg", anonymous=True)

    topic: str = rospy.get_param("~topic", default="/move_base/global_costmap/costmap")

    sub = rospy.Subscriber(topic, numpy_msg(OccupancyGrid), queue_size=1, callback=callback)
    print("created the sub")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
