#! /usr/bin/python3
import rospy


def main():
    rospy.init_node("map_server")

    topic: str = '/map'
    data_dir: str = ''

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
