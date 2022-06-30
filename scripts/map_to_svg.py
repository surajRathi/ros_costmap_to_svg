#! /usr/bin/python3
import cv2
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg


def callback(msg: OccupancyGrid):
    rospy.loginfo("Received a cost map")
    map_info = msg.info

    img = msg.data.reshape(map_info.height, map_info.width).astype(np.uint8)

    thresh_1 = 30
    thresh_2 = 80
    assert thresh_1 <= thresh_2

    print((img <= thresh_1).sum(), ((img <= thresh_2) & (img > thresh_1)).sum(), (img > thresh_2).sum(), )

    rgb = np.empty((img.shape[0], img.shape[1], 3), np.uint8)
    rgb[img <= thresh_1, ...] = (255, 0, 0)
    rgb[(img <= thresh_2) & (img > thresh_1), ...] = (0, 255, 0)
    rgb[(img > thresh_2), ...] = (0, 0, 255)

    contours, hierarchy = cv2.findContours((img <= thresh_1).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_TC89_L1

    print(f"got {len(contours)} contours")
    cv2.drawContours(rgb, contours, -1, color=(255, 255, 255), thickness=2)

    cv2.imshow("msg", rgb)
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
