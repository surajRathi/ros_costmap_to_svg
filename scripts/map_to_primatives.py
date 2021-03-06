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
    scale: Optional[float] = None

    def update(self):
        if self.img is None:
            return

        # edges = cv2.bitwise_and((self.img == 255).astype('uint8') * 255,
        edges = cv2.dilate((self.img == 0).astype('uint8') * 255, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), 1) \
                & ((self.img == 100).astype('uint8') * 255)

        lines = cv2.HoughLinesP(edges,
                                rho=max(1, int(5 / self.scale)), theta=np.pi / 180,
                                threshold=max(1, int(100 / self.scale / self.scale)),
                                minLineLength=max(1, int(60 / self.scale)),
                                maxLineGap=max(1, int(20 / self.scale))
                                )
        thickness = max(1, int(20 / self.scale))

        line_mask = np.zeros_like(edges)
        for (l,) in lines:
            cv2.line(line_mask, (l[0], l[1]), (l[2], l[3]), (255,), thickness=thickness, lineType=cv2.LINE_8)
        # cv2.dilate  # The thickness kind of dilates it already, can we leave this?
        print(self.img.max(), self.img.min(), (self.img == 0).sum(), (self.img == 100).sum(), (self.img == 255).sum())
        obs = ((self.img == 100).astype('uint8') * 255) & ~line_mask

        # circles = cv2.HoughCircles(obs, method=cv2.HOUGH_GRADIENT,
        #                            # dp=max(1, (1.5 * 5 / self.scale)),
        #                            dp=1.0,
        #                            minDist=max(1, int(25 / self.scale)),
        #                            minRadius=max(1, int(10 / self.scale)),
        #                            param1=50, param2=30)
        # print(circles)
        # if circles is None:
        #     circles = []
        # for c in circles:
        #     color = tuple(map(int, np.random.randint(0, 256, 3)))
        #     cv2.circle(obs, (c[0], c[1]), c[2], color, thickness)
        #     # draw the center of the circle
        #     cv2.circle(out, (c[0], c[1]), 2, (0, 0, 255), 3)
        # # Remove circles from obs mask

        obs = cv2.dilate(obs, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), 1)

        out = np.repeat(self.img[..., np.newaxis], 3, axis=-1)
        for (l,) in lines:
            color = tuple(map(int, np.random.randint(0, 256, 3)))
            cv2.line(out, (l[0], l[1]), (l[2], l[3]), color, thickness=thickness, lineType=cv2.LINE_AA)

        # for c in circles:
        #     color = tuple(map(int, np.random.randint(0, 256, 3)))
        #     cv2.circle(out, (c[0], c[1]), c[2], color, thickness)
        #     # draw the center of the circle
        #     cv2.circle(out, (c[0], c[1]), 2, (0, 0, 255), 3)

        cv2.imshow('lines', cv2.resize(out, (int(out.shape[1] * self.scale), int(out.shape[0] * self.scale)),
                                       interpolation=cv2.INTER_NEAREST))
        cv2.imshow('obs', cv2.resize(obs, (int(obs.shape[1] * self.scale), int(obs.shape[0] * self.scale)),
                                     interpolation=cv2.INTER_NEAREST))
        # cv2.imshow('aa', (self.img == 100).astype('uint8') * 255)
        cv2.waitKey(20)

    def callback(self, msg: numpy_msg(OccupancyGrid)):
        rospy.loginfo("Received a cost map")
        if msg.info.resolution < 0.01 - 0.001:
            rospy.logerr("Package is designed for maps with a resolution greater than 0.01 only. Failing")
            return
        self.img = msg.data.reshape(msg.info.height, msg.info.width).astype(np.uint8)
        self.scale = max(1, msg.info.resolution / 0.01)
        rospy.loginfo(f"{self.scale=}")
        self.update()

    def callback_updates(self, msg: numpy_msg(OccupancyGridUpdate)):
        rospy.logdebug("Received a cost map update")
        if self.img is None:
            rospy.logwarn("Received update before receiving the cost map")
            return

        # Note: Axes are flipped
        self.img[msg.y:msg.y + msg.height, msg.x:msg.x + msg.width] = msg.data.reshape(msg.height, msg.width).astype(
            np.uint8)

        self.update()


def main():
    rospy.init_node("map_to_svg", anonymous=True)

    topic: str = '/map'

    data = CommonData(pub=rospy.Publisher(topic + '_svg', String, queue_size=1, latch=True))
    sub = rospy.Subscriber(topic, numpy_msg(OccupancyGrid), queue_size=1, callback=data.callback)
    # sub_updates = rospy.Subscriber(topic + "_updates", numpy_msg(OccupancyGridUpdate), queue_size=1,
    #                                callback=data.callback_updates)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
