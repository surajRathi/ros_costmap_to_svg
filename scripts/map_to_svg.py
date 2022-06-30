#! /usr/bin/python3
import rospy

from nav_msgs.msg import OccupancyGrid


def callback(msg: OccupancyGrid):
    print(msg.header)
    print(msg.data)


    map_info = msg.info

    rospy.signal_shutdown("work done")


# def process_image()
def main():
    rospy.init_node("map_to_svg", anonymous=True)

    topic: str = rospy.get_param("~topic", default="/move_base/local_costmap/costmap")

    sub = rospy.Subscriber(topic, OccupancyGrid, queue_size=1, callback=callback)
    print("created the sub")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
