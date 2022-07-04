#! /usr/bin/python3
import dataclasses
import io
from typing import Optional

import cv2
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String


@dataclasses.dataclass
class CommonData:
    pub: rospy.Publisher
    thresh: int = 20
    erosion_iter: int = 2
    # https://stackoverflow.com/a/61099329/1515394
    fill_color: str = f"rgb(0, 0, {int(255 * 0.8)})"
    fill_opacity: str = '0.4'
    img: Optional[np.ndarray] = None

    def update(self):
        if self.img is None:
            return

        thresh = 80
        img = (self.img >= thresh).astype(np.uint8)
        img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), self.erosion_iter)
        img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), self.erosion_iter)
        img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), self.erosion_iter)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        rospy.logdebug(f"Got {len(contours)} contours")

        with io.StringIO() as f:
            f.write(
                f"<svg width='{self.img.shape[0]}' height='{self.img.shape[1]}' viewbox='0 0 {self.img.shape[0]} {self.img.shape[1]}' "
                "fill='#044B94' fill-opacity='0.4' xmlns='http://www.w3.org/2000/svg' >")
            f.write(f"<path style='stroke-width:0px' fill-opacity='{self.fill_opacity}' fill='{self.fill_color}' d='")
            for contour in contours:

                f.write(f" M {contour[0][0][0]} {contour[0][0][1]}")
                for (x, y), in contour[1:]:
                    f.write(f"L {x} {y} ")
                f.write(f"Z ")

            f.write("' />")
            f.write(f"</svg>")

            f.seek(0)
            self.pub.publish(f.read())


def callback(msg: numpy_msg(OccupancyGrid), c: CommonData):
    rospy.loginfo("Received a cost map")
    c.img = msg.data.reshape(msg.info.height, msg.info.width).astype(np.uint8)
    c.update()


def callback_updates(msg: numpy_msg(OccupancyGridUpdate), c: CommonData):
    rospy.logdebug("Received a cost map update")
    if c.img is None:
        rospy.logwarn("Received update before receiving the cost map")
        return

    # Note: Axes are flipped
    c.img[msg.y:msg.y + msg.height, msg.x:msg.x + msg.width] = msg.data.reshape(msg.height, msg.width).astype(np.uint8)

    # cv2.imshow('aa', c.img)
    # cv2.waitKey(1)
    c.update()


# def process_image()
def main():
    rospy.init_node("map_to_svg", anonymous=True)

    g_topic: str = '/move_base/global_costmap/costmap'

    g_data = CommonData(pub=rospy.Publisher(g_topic + '_svg', String, queue_size=1, latch=True))
    g_data.fill_color = f"rgb(0, 0, {int(255 * 0.8)})"
    g_sub = rospy.Subscriber(g_topic, numpy_msg(OccupancyGrid), queue_size=1, callback=callback, callback_args=g_data)
    g_sub_updates = rospy.Subscriber(g_topic + "_updates", numpy_msg(OccupancyGridUpdate), queue_size=1,
                                     callback=callback_updates,
                                     callback_args=g_data)

    topic: str = '/move_base/local_costmap/costmap'

    data = CommonData(pub=rospy.Publisher(topic + '_svg', String, queue_size=1, latch=True))
    data.fill_color = f"rgb({int(255 * 0.8)}, 0, 0)"
    sub = rospy.Subscriber(topic, numpy_msg(OccupancyGrid), queue_size=1, callback=callback, callback_args=data)
    sub_updates = rospy.Subscriber(topic + "_updates", numpy_msg(OccupancyGridUpdate), queue_size=1,
                                   callback=callback_updates,
                                   callback_args=data)
    print("created the sub")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
