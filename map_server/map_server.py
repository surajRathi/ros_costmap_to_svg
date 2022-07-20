#! /usr/bin/python3
import base64
import io
import pathlib
from typing import Optional, Tuple, Callable

import numpy as np
import rospy
import yaml
from PIL import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rospy.numpy_msg import numpy_msg
from map_to_svg.srv import SetMap, SetMapRequest, SetMapResponse

"""
Contents of a map directory:
- map.yaml [REQUIRED]
- map.pgm
- map.png
- map.svg
- map.lock [MAYBE]

Either one of `map.pgm` and `map.svg` must exist.
"""


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


def read_yaml(path: pathlib.Path) -> Optional[Tuple[MapMetaData, Callable[[np.ndarray], np.ndarray]]]:
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

            negate = bool(data['negate'])
            occupied_thresh = float(data['occupied_thresh'])
            free_thresh = float(data['free_thresh'])

            mode = data['mode'] if 'mode' in data else 'trinary'
            if mode not in ('trinary', 'scale', 'raw'):
                rospy.logerr(f"Invalid mode {mode}, should be one of 'trinary', 'scale', or 'raw'.")

            def value_interpreter(arr: np.ndarray):
                if not negate:
                    arr = 255 - arr

                if mode == 'trinary':
                    out = np.zeros_like(arr).astype(np.uint8)
                    out[...] = 255
                    out[arr > (occupied_thresh * 255)] = 100
                    out[arr < (free_thresh * 255)] = 0
                    return out
                if mode == 'scale':
                    out = np.zeros_like(arr).astype(np.uint8)
                    out[...] = 255
                    out[arr > occupied_thresh * 255] = 100
                    out[arr < free_thresh * 255] = 0
                    out[out == 255] = (
                            99 * (arr[out == 255] - free_thresh * 255) / (255 * (occupied_thresh - free_thresh))
                    ).astype(np.uint8)

                    return out
                if mode == 'raw':
                    return arr

                assert False

            return m, value_interpreter

        except yaml.YAMLError as exc:
            rospy.logerr(f"Could not read the YAML file {path}, {exc}")
            return None
        except KeyError as e:
            rospy.logerr(f"The YAML file at {path} is missing the {e} key.")


def read_pgm(file: pathlib.Path) -> Optional[np.ndarray]:
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
            return arr

        except None:
            print("Error")
            return None


class MapPublisher:
    def __init__(self, frame_id: str, data_dir: str, name: Optional[str] = None):
        self.loaded = False
        self.data: Optional[np.ndarray] = None
        self.meta_data: Optional[MapMetaData] = None
        self.map_data: Optional[numpy_msg(OccupancyGrid)] = None

        self.frame_id = frame_id
        self.data_dir = data_dir
        self.name = name

        self.meta_pub = rospy.Publisher('map_metadata', data_class=MapMetaData, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('map', data_class=numpy_msg(OccupancyGrid), queue_size=1, latch=True)

        self.set_map_srv = rospy.Service('set_map_srv', SetMap, self.set_map)
        if self.name is not None:
            self.loaded = self.load()

    def set_map(self, req: SetMapRequest) -> SetMapResponse:
        resp = SetMapResponse()

        if req.map == '':
            resp.set_map = '' if self.name is None else self.name
        else:
            self.name = req.map
            self.loaded = self.load()
            if not self.loaded:
                self.name = None
            resp.set_map = '' if self.name is None else self.name

        return resp

    def load(self) -> bool:  # Success
        if self.name is None:
            return False
        if not check_map_dir(self.name, self.data_dir):
            return False

        self.meta_data, value_interpreter = read_yaml(pathlib.Path(self.data_dir) / self.name / 'map.yaml')
        if self.meta_data is None:
            rospy.logerr("Could not load the yaml file.")
            return False

        map_arr = read_pgm(pathlib.Path(self.data_dir) / self.name / 'map.pgm')
        if map_arr is None:
            rospy.logerr("Could not load the pgm file.")
            return False

        self.meta_data.width, self.meta_data.height = map_arr.shape

        self.map_data: numpy_msg(OccupancyGrid) = numpy_msg(OccupancyGrid)()
        # self.map_data: OccupancyGrid = numpy_msg(OccupancyGrid)()  # Uncomment for autocomplete
        self.map_data.header.frame_id = self.frame_id
        self.map_data.header.stamp = self.meta_data.map_load_time
        self.map_data.info = self.meta_data
        self.map_data.data = value_interpreter(map_arr.reshape(-1)).astype(np.uint8)

        # print(self.map_data, np.unique(self.map_data.data), sep='\n')

        self.meta_pub.publish(self.meta_data)
        self.map_pub.publish(self.map_data)
        rospy.loginfo('Publishing to the topic map and map_metadata.')
        return True

    def as_base64_svg_editor_bg(self) -> str:
        if not self.loaded:
            return ''

        # Returns a base64 encoded png of the map
        # The PNG is flipped in the x (rows) axis.
        # Only implemented for trinary

        image = np.zeros((self.meta_data.width, self.meta_data.height, 3), dtype=np.uint8)
        image[...] = (93, 100, 108)  # for im_like == -1
        im_like = self.map_data.data.reshape(self.meta_data.width, self.meta_data.height)
        image[im_like[::-1, :] == 100] = (14, 14, 14)
        image[im_like[::-1, :] == 0] = (193, 193, 193)

        with io.BytesIO() as f:
            Image.fromarray(image).save(f, 'PNG')
            f.seek(0)
            return base64.b64encode(f.read()).decode('ASCII')


def main():
    rospy.init_node("map_server")

    data_dir: str = '/home/suraj/ws/src/rosjs/map_to_svg/data/final'
    frame_id: str = rospy.get_param('~frame_id', 'map')

    m = MapPublisher(frame_id, data_dir, 'tb3')
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
