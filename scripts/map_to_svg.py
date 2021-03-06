#! /usr/bin/python3
import dataclasses
import io
from typing import Optional, List

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
    thresh: int = 50
    erosion_iter: int = 2
    erosion_size: int = 4
    # https://stackoverflow.com/a/61099329/1515394
    fill_color: str = f"rgb(0, 0, {int(255 * 0.8)})"
    fill_opacity: str = '0.4'
    map_walls_fill_color: str = f"rgb(0, 0, {int(255 * 0.8)})"
    map_walls_fill_opacity: str = '0.4'

    img: Optional[np.ndarray] = None

    def update(self):
        if self.img is None:
            return

        img = (self.img >= self.thresh).astype(np.uint8)
        erosion_shape = (self.erosion_size, self.erosion_size)

        if self.erosion_iter != 0:
            img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_shape), self.erosion_iter)
            img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_shape), self.erosion_iter)
            img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_shape), self.erosion_iter)

        contours: List[np.ndarray]
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        rospy.logdebug(f"Got {len(contours)} contours")

        with io.StringIO() as f:
            f.write(
                f"<svg width='{self.img.shape[0]}' height='{self.img.shape[1]}' viewbox='0 0 {self.img.shape[0]} {self.img.shape[1]}' "
                "fill='#044B94' fill-opacity='0.4' xmlns='http://www.w3.org/2000/svg' >")
            f.write(f"<path style='stroke-width:0px' fill-opacity='{self.fill_opacity}' fill='{self.fill_color}' "
                    f"stroke='none' "
                    # f"stroke='{self.fill_color}' stroke-width='10' stroke-opacity='{self.fill_opacity}' stroke-linejoin='round' stroke-linecap='round' "
                    f"d='"
                    )
            for contour in contours:
                f.write(f" M {contour[0][0][0]} {contour[0][0][1]} ")
                for ((xo, yo),), ((x, y),), ((xn, yn),) in zip(contour[:-1], contour[1:], np.roll(contour, -1)[1:]):
                    th_o = np.arctan2(yo - y, xo - x)
                    th_r = np.arctan2(y - yn, x - xn)
                    th_t = 0.8 * th_o + th_r * 0.2
                    # l = 0.1 * min(np.sqrt((x - xo) ** 2 + (y - yo) ** 2), np.sqrt((x - xn) ** 2 + (y - yn) ** 2))
                    # l = 0.3 * np.sqrt((x - xo) ** 2 + (y - yo) ** 2) # , np.sqrt((x - xn) ** 2 + (y - yn) ** 2))
                    l = 2
                    xc = x + l * np.cos(th_t)
                    yc = y + l * np.sin(th_t)
                    # print(xc, yc, th_t)
                    f.write(f"S {int(xc)} {int(yc)}, {x} {y} ")
                f.write(f"Z ")

            f.write("' />")

            # for contour in contours:
            #     for (x, y), in contour:
            #         f.write(
            #             f"<circle cx=\"{x}\" cy=\"{y}\" r=\"1\" stroke=\"none\" stroke-width=\"0\" fill=\"black\" />")
            #
            # for contour in contours:
            #     for ((xo, yo),), ((x, y),), ((xn, yn),) in zip(contour[:-1], contour[1:], np.roll(contour, -1)[1:]):
            #         th_o = np.arctan2(yo - y, xo - x)
            #         th_r = np.arctan2(y - yn, x - xn)
            #         th_t = 0.8 * th_o + th_r * 0.2
            #         # l = 0.1 * min(np.sqrt((x - xo) ** 2 + (y - yo) ** 2), np.sqrt((x - xn) ** 2 + (y - yn) ** 2))
            #         # l = 0.3 * np.sqrt((x - xo) ** 2 + (y - yo) ** 2) # , np.sqrt((x - xn) ** 2 + (y - yn) ** 2))
            #         l = 2
            #         xc = x + l * np.cos(th_t)
            #         yc = y + l * np.sin(th_t)
            #         f.write(
            #             f"<circle cx=\"{int(xc)}\" cy=\"{int(yc)}\" r=\"1\" stroke=\"none\" stroke-width=\"0\" fill=\"green\" />")

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

    topic: str = '/move_base/local_costmap/costmap'

    data = CommonData(pub=rospy.Publisher(topic + '_svg', String, queue_size=1, latch=True))
    data.fill_color = f"rgb({int(255 * 0.8)}, 0, 0)"
    data.fill_opacity = "0.6"
    data.thresh = 60
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
