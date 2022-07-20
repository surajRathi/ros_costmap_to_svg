#! /usr/bin/python3
import pathlib
from typing import Optional, Tuple
import yaml

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData

"""
Contents of a map directory:
- map.yaml [REQUIRED]
- map.pgm
- map.png
- map.svg
- map.lock [MAYBE]

Either one of `map.pgm` and `map.svg` must exist.
"""


def read_pgm(file: pathlib.Path) -> Optional[Tuple[np.ndarray, int]]:
    if not file.is_file():
        return None

    with file.open('rb') as f:
        # https://users.wpi.edu/~cfurlong/me-593n/pgmimage.html
        try:
            p5, comment, dims, max_val = f.readline(), f.readline(), f.readline(), f.readline()
            if p5 != b"P5\n":
                rospy.logerr("Invlid PGM File")
                return None
            rows, cols = map(int, dims.decode('ASCII').strip().split(' '))
            maximum_value = int(max_val.decode('ASCII').strip())
            if maximum_value > 255:
                rospy.logerr(f"No support for reading pgm files with max value greater than {maximum_value}")
                return None

            map_data = f.read()
            if len(map_data) != rows * cols:
                rospy.logerr(f"Invalid PGM file read {len(map_data)}, expected {rows * cols}.")
                return None

            arr: np.ndarray = (np.frombuffer(map_data, dtype=np.uint8)).reshape(rows, cols)
            return arr, maximum_value

        except None:
            print("Error")
            return None


def check_map_dir(name: str, data_dir: str) -> bool:
    base = pathlib.Path(data_dir)
    if not base.exists():
        rospy.logerr(f"{base} does not exist")
        return False

    if not base.is_dir():
        rospy.logerr(f"Data directory {base} is not a directory.")
        return False

    map_dir = base / name
    if not map_dir.exists():
        rospy.logerr(f"{map_dir} does not exist")
        return False

    if not map_dir.is_dir():
        rospy.logerr(f"Map directory {map_dir} is not a directory.")
        return False

    yaml_path = map_dir / 'map.yaml'
    if (not yaml_path.exists()) or yaml_path.is_dir():
        rospy.logerr(f"Invalid yaml path: {yaml_path}")
        return False

    pgm_path = map_dir / 'map.pgm'
    if not pgm_path.exists():
        if (map_dir / 'map.svg').exists():
            rospy.logerr("SVG to pgm conversion is not implemented yet. Failing")
            # TODO: Convert SVG to pgm.
            return False
        if pgm_path.is_dir():
            rospy.logerr(f"PGM {pgm_path} is a directory.")
            return False
    return True


def load_yaml(path: pathlib.Path) -> Optional[MapMetaData]:
    with path.open('r') as f:
        try:
            data = yaml.safe_load(f)
            if data['image'] != 'map.pgm':
                rospy.logerr(f"Field image in map.yaml must be map.pgm and not {data['image']}")
                return None
            m = MapMetaData()
            m.map_load_time = rospy.Time.now()
            m.resolution = data['resolution']
            m.origin = Pose()
            m.origin.position.x, m.origin.position.y, m.origin.position.z = data['origin']
            m.origin.orientation.w = 1
            return m

        except yaml.YAMLError as exc:
            rospy.logerr(f"Could not read the YAML file {path}, {exc}")
            return None


class MapPublisher:
    def __init__(self, topic: str, name: str, data_dir: str):
        self.is_init = False
        self.data: Optional[np.ndarray] = None
        self.max_val: Optional[int] = None
        self.meta_data: Optional[MapMetaData] = None

        if self.init(name, data_dir):
            self.is_init = True

    def init(self, name: str, data_dir: str) -> bool:  # Success
        if not check_map_dir(name, data_dir):
            return False

        self.meta_data = load_yaml(pathlib.Path(data_dir) / name / 'map.yaml')
        if self.meta_data is None:
            rospy.logerr("Could not load the yaml file.")
            return False

        read_pgm_ret = read_pgm(pathlib.Path(data_dir) / name / 'map.pgm')
        if read_pgm_ret is None:
            rospy.logerr("Could not load the pgm file.")
            return False

        self.data, self.max_val = read_pgm_ret
        self.meta_data.width, self.meta_data.height = self.data.shape
        return True

    def update_map(self):
        pass


def main():
    rospy.init_node("map_server")

    topic: str = '/map'
    data_dir: str = '/home/suraj/ws/src/rosjs/map_to_svg/data/final'

    m = MapPublisher(topic, 'tb3', data_dir)
    # rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
