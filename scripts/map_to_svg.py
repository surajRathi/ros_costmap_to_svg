#! /usr/bin/python3
import io

import cv2
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String

svg_filename = "out.svg"


def callback(msg: OccupancyGrid, pub: rospy.Publisher):
    rospy.loginfo("Received a cost map")
    map_info = msg.info

    img = msg.data.reshape(map_info.height, map_info.width).astype(np.uint8)

    thresh = 80

    contours, hierarchy = cv2.findContours(
        cv2.erode((img >= thresh).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_TC89_L1)

    print(f"got {len(contours)} contours")

    with io.StringIO() as f:
        f.write(f"<svg width='{img.shape[0]}' height='{img.shape[1]}' viewbox='0 0 {img.shape[0]} {img.shape[1]}' "
                "fill='#044B94' fill-opacity='0.4' xmlns='http://www.w3.org/2000/svg' >")
        for contour in contours:
            f.write(
                f"<path style='fill:none;stroke:#00FFFF44;stroke-width:2px;stroke-opacity:1' stroke-linejoin='round'")
            f.write(f" d='M {contour[0][0][0]} {contour[0][0][1]}")
            for (x, y), in contour[1:]:
                f.write(f"L {x} {y} ")
            f.write(f"L {contour[0][0][0]} {contour[0][0][1]} ")

            f.write("' />")
        f.write(f"</svg>")

        f.seek(0)
        pub.publish(f.read())


# def process_image()
def main():
    rospy.init_node("map_to_svg", anonymous=True)

    topic: str = rospy.get_param("~topic", default="/move_base/global_costmap/costmap")
    pub_topic: str = rospy.get_param("~pub_topic", default="/move_base/global_costmap/costmap_svg")

    pub = rospy.Publisher(pub_topic, String, queue_size=1, latch=True)
    sub = rospy.Subscriber(topic, numpy_msg(OccupancyGrid), queue_size=1, callback=callback, callback_args=pub)
    print("created the sub")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
