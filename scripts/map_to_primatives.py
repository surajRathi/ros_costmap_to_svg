#! /usr/bin/python3
import dataclasses
from typing import Optional

import cv2
import numpy as np
import rospy
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String


@dataclasses.dataclass
class CommonData:
    pub: rospy.Publisher

    img: Optional[np.ndarray] = None

    def update(self):
        if self.img is None:
            return

        # edges = cv2.bitwise_and((self.img == 255).astype('uint8') * 255,
        edges = cv2.dilate((self.img == 0).astype('uint8') * 255, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), 1) \
                & self.img

        out = np.repeat(self.img[..., np.newaxis], 3, axis=-1)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 15)
        for (l,) in lines:
            # color = tuple((np.random.random((3,)) * 255).astype(np.uint8))
            color = (0, 255, 0)
            cv2.line(out, (l[0], l[1]), (l[2], l[3]),
                     color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('aa', edges)
        cv2.imshow('aa', out)
        cv2.waitKey(20)


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

    topic: str = '/map'

    data = CommonData(pub=rospy.Publisher(topic + '_svg', String, queue_size=1, latch=True))
    sub = rospy.Subscriber(topic, numpy_msg(OccupancyGrid), queue_size=1, callback=callback, callback_args=data)
    # sub_updates = rospy.Subscriber(topic + "_updates", numpy_msg(OccupancyGridUpdate), queue_size=1,
    #                                callback=callback_updates,
    #                                callback_args=data)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
