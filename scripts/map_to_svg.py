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

svg_filename = "out.svg"


@dataclasses.dataclass
class CommonData:
    pub: rospy.Publisher
    thresh: int = 80
    stroke_color: str = '#00FFFF66'
    # https://stackoverflow.com/a/61099329/1515394
    fill_color: str = 'rgb(255, 0, 0)'  # '#00FFFF66'.replace('#', '%23')
    fill_opacity: str = '0.4'
    img: Optional[np.ndarray] = None

    def update(self):
        if self.img is None:
            return

        # cv2.imshow('map', self.img)
        # cv2.waitKey(0)

        thresh = 80
        contours, hierarchy = cv2.findContours(
            cv2.erode((self.img >= thresh).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_TC89_L1)

        rospy.logdebug(f"Got {len(contours)} contours")

        with io.StringIO() as f:
            f.write(
                f"<svg width='{self.img.shape[0]}' height='{self.img.shape[1]}' viewbox='0 0 {self.img.shape[0]} {self.img.shape[1]}' "
                "fill='#044B94' fill-opacity='0.4' xmlns='http://www.w3.org/2000/svg' >")
            f.write(
                f"<path style='stroke:{self.stroke_color};stroke-width:2px;stroke-opacity:1' "
                f"stroke-linecap='round' stroke-linejoin='round' "
                f"fill-opacity='{self.fill_opacity}' fill='{self.fill_color}' d='")
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

    topic: str = rospy.get_param("~topic", default="/move_base/global_costmap/costmap")
    pub_topic: str = rospy.get_param("~pub_topic", default="/move_base/global_costmap/costmap_svg")

    c = CommonData(pub=rospy.Publisher(pub_topic, String, queue_size=1, latch=True))
    sub = rospy.Subscriber(topic, numpy_msg(OccupancyGrid), queue_size=1, callback=callback, callback_args=c)
    sub_updates = rospy.Subscriber(topic + "_updates", numpy_msg(OccupancyGridUpdate), queue_size=1,
                                   callback=callback_updates,
                                   callback_args=c)
    print("created the sub")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
