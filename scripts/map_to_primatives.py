#! /usr/bin/python3
import dataclasses
from typing import Optional

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
        pass


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
