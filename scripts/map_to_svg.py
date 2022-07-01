#! /usr/bin/python3
import time

import cv2
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg

svg_filename = "out.svg"


def callback(msg: OccupancyGrid):
    rospy.loginfo("Received a cost map")
    map_info = msg.info

    img = msg.data.reshape(map_info.height, map_info.width).astype(np.uint8)

    thresh_1 = 30
    thresh_2 = 80
    assert thresh_1 <= thresh_2

    print((img <= thresh_1).sum(), ((img <= thresh_2) & (img > thresh_1)).sum(), (img > thresh_2).sum(), )

    rgb = np.empty((img.shape[0], img.shape[1], 3), np.uint8)
    rgb[img <= thresh_1, ...] = (0, 0, 0)
    rgb[(img <= thresh_2) & (img > thresh_1), ...] = (0, 128, 0)
    rgb[(img > thresh_2), ...] = (0, 128, 0)

    contours, hierarchy = cv2.findContours((img <= thresh_2).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_TC89_L1

    print(f"got {len(contours)} contours")
    # cv2.drawContours(rgb, contours, -1, color=(255, 255, 255), thickness=2)
    # scale = 8
    # rgb_up = cv2.resize(rgb, (img.shape[0] * scale, img.shape[1] * scale), interpolation=cv2.INTER_NEAREST)
    # rgb_up = cv2.blur(rgb_up, (3 * 8, 3 * 8))
    # rgb_up[rgb_up < 127] = 0
    # rgb_up[rgb_up >= 127] = 127
    #
    # cv2.imshow('msg', rgb_up)
    # cv2.waitKey()
    # cv2.destroyWindow('msg')
    with open(svg_filename, 'w') as f:
        f.write(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{img.shape[0]}' height='{img.shape[1]}' fill='#044B94' fill-opacity='0.4'>")
        for contour in contours:
            f.write(f"<path style='fill:none;stroke:#AAAAAA;stroke-width:2px;stroke-opacity:1' ")
            f.write(f" d='M {contour[0][0][0]} {contour[0][0][1]}")
            print(len(contour), len(contour[0][0]))
            for (x, y), in contour[1:]:
                f.write(f"L {x} {y} ")
            f.write(f"L {contour[0][0][0]} {contour[0][0][1]} ")

            f.write("' />")
        f.write(f"</svg>")
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
