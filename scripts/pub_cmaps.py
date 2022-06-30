#! /usr/bin/python3
import rospy
import rosbag
from nav_msgs.msg import OccupancyGrid

pkg_name = 'map_to_svg'
cmap_topic = '/move_base/global_costmap/costmap'


def main():
    import rospkg
    try:
        pkg_path = rospkg.RosPack().get_path(pkg_name)
    except rospkg.ResourceNotFound:
        print('cant find the package')
        return

    cmap = None
    for topic, msg, t in rosbag.Bag(f"{pkg_path}/data/base_cmaps.bag").read_messages(
            topics=(cmap_topic,)):
        cmap = msg

    if cmap is None:
        print('cant get cmap msg')
        return

    rospy.init_node('cmap_pub')
    pub = rospy.Publisher(cmap_topic, OccupancyGrid, queue_size=1, latch=True)
    pub.publish(cmap)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
