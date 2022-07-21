#! /usr/bin/python3
import base64
import io
import pathlib
import time
from typing import Optional, Tuple, Callable

import cv2
import numpy as np
import rospy
import yaml
from PIL import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rospy.numpy_msg import numpy_msg
from nav_msgs.srv import GetMap, GetMapRequest, GetMapResponse
from map_to_svg.srv import SetMap, SetMapRequest, SetMapResponse
from map_to_svg.srv import StartEditing, StartEditingRequest, StartEditingResponse
from map_to_svg.srv import FinishEditing, FinishEditingRequest, FinishEditingResponse
from map_to_svg.srv import ListMaps, ListMapsRequest, ListMapsResponse

"""
Contents of a map directory:
- map.yaml [REQUIRED]
- map.pgm
- map.png
- map.svg
- map.lock [MAYBE]

Either one of `map.pgm` and `map.svg` must exist.
"""
im = None


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

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, new_name: Optional[str]):
        self.loaded = False
        if new_name is not None:
            self.loaded = self.load(new_name)

        self._name = new_name

        if not self.loaded:
            # self.meta_data = MapMetaData()
            # self.map_data = numpy_msg(OccupancyGrid)()
            # self.map_data.info = self.meta_data
            # self.map_data.data = np.zeros((0,))
            #
            # self.meta_pub.publish(self.meta_data)
            # self.map_pub.publish(self.map_data)
            rospy.logerr('Invalid map name')
            self._name = None

    def __init__(self, frame_id: str, data_dir: str, name: Optional[str] = None):
        self.loaded = False
        self.data: Optional[np.ndarray] = None
        self.meta_data: Optional[MapMetaData] = None
        self.map_data: Optional[numpy_msg(OccupancyGrid)] = None

        self._name = None

        self.frame_id = frame_id
        self.data_dir = data_dir

        self.meta_pub = rospy.Publisher('map_metadata', data_class=MapMetaData, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('map', data_class=numpy_msg(OccupancyGrid), queue_size=1, latch=True)

        self.static_map_srv = rospy.Service('static_map', GetMap, self.get_map)  # TODO: Check
        self.list_maps_srv = rospy.Service('map_server/list_maps', ListMaps, self.list_maps)
        self.set_map_srv = rospy.Service('map_server/set_map', SetMap, self.set_map)
        self.start_editing_srv = rospy.Service('map_server/start_editing', StartEditing, self.start_editing)
        self.finish_editing_srv = rospy.Service('map_server/finish_editing', FinishEditing, self.finish_editing)

        self.name = name

    def get_map(self, req: GetMapRequest) -> GetMapResponse:
        resp = GetMapResponse()
        resp.map = self.map_data
        return resp

    def list_maps(self, req: ListMapsRequest) -> ListMapsResponse:
        resp = ListMapsResponse()
        resp.maps = []
        for file in pathlib.Path(self.data_dir).iterdir():
            if file.is_dir() and check_map_dir(file.name, self.data_dir):
                resp.maps.append(file.name)

        resp.current = '' if self.name is None else self.name
        return resp

    def set_map(self, req: SetMapRequest) -> SetMapResponse:
        if req.map != '':
            self.name = req.map

        resp = SetMapResponse()
        resp.set_map = '' if self.name is None else self.name

        return resp

    def start_editing(self, req: StartEditingRequest) -> StartEditingResponse:
        if not self.loaded:
            resp = StartEditingResponse()
            return resp
        svg_file = pathlib.Path(self.data_dir) / self.name / 'map.svg'
        if not svg_file.exists():
            self.create_svg()
        resp = StartEditingResponse()
        resp.success = 1
        resp.svg_data = svg_file.open('r').read()
        resp.raw_png = self.as_base64_svg_editor_bg()
        return resp

    def finish_editing(self, req: FinishEditingRequest) -> FinishEditingResponse:
        print(req.svg_data)
        return FinishEditingResponse()

    def load(self, name) -> bool:  # Success
        if name is None:
            return False
        if not check_map_dir(name, self.data_dir):
            return False

        self.meta_data, value_interpreter = read_yaml(pathlib.Path(self.data_dir) / name / 'map.yaml')
        if self.meta_data is None:
            rospy.logerr("Could not load the yaml file.")
            return False

        map_arr = read_pgm(pathlib.Path(self.data_dir) / name / 'map.pgm')
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

        self.create_svg()
        return True

    def as_base64_svg_editor_bg(self) -> str:
        if not self.loaded:
            return ''

        # Returns a base64 encoded png of the map

        image = np.zeros((self.meta_data.height, self.meta_data.width, 3), dtype=np.uint8)
        image[...] = (93, 100, 108)  # for im_like == -1
        im_like = self.map_data.data.reshape(self.meta_data.height, self.meta_data.width)
        image[im_like == 100] = (14, 14, 14)
        image[im_like == 0] = (193, 193, 193)

        global im
        im = image.copy()

        with io.BytesIO() as f:
            Image.fromarray(image).save(f, 'PNG')
            f.seek(0)
            return base64.b64encode(f.read()).decode('ASCII')

    def create_svg(self):
        if not self.loaded:
            return ''
        scale = max(1, self.map_data.info.resolution / 0.01)
        img = self.map_data.data.reshape(self.meta_data.width, self.meta_data.height)

        edges = cv2.dilate((img == 0).astype('uint8') * 255, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), 1) \
                & ((img == 100).astype('uint8') * 255)

        lines = cv2.HoughLinesP(edges,
                                rho=max(1, int(5 / scale)), theta=np.pi / 180,
                                threshold=max(1, int(100 / scale / scale)),
                                minLineLength=max(1, int(60 / scale)),
                                maxLineGap=max(1, int(20 / scale))
                                )

        thickness = max(1, int(20 / scale))
        line_mask = np.zeros_like(edges)
        for (l,) in lines:
            cv2.line(line_mask, (l[0], l[1]), (l[2], l[3]), (255,), thickness=thickness, lineType=cv2.LINE_8)
        # cv2.dilate  # The thickness kind of dilates it already, can we leave this?
        # print(img.max(), img.min(), (img == 0).sum(), (img == 100).sum(), (img == 255).sum())
        obs = ((img == 100).astype('uint8') * 255) & ~line_mask

        with (pathlib.Path(self.data_dir) / self.name / 'map.svg').open('w') as f:
            f.write(
                f"<svg xmlns='http://www.w3.org/2000/svg' preserveAspectRatio='xMinYMin meet' viewBox='0 0 {img.shape[0]} {img.shape[1]}'>\n")

            f.write(f"<rect x='0' y='0' class='bg' width='{img.shape[0]}' height='{img.shape[1]}' />\n")

            for (l,) in lines:
                f.write(f"<line x1='{l[0]}' y1='{l[1]}' x2='{l[2]}' y2='{l[3]}' class='item obstacle' />\n")

            f.write("</svg>")


def main():
    rospy.init_node("map_server")

    data_dir: str = '/home/suraj/ws/src/rosjs/map_to_svg/data/final'
    frame_id: str = rospy.get_param('~frame_id', 'map')

    m = MapPublisher(frame_id, data_dir, 'tb3_working')
    import matplotlib.pyplot as plt
    global im
    while True:
        if im is not None:
            plt.imshow(im)
            plt.pause(1)
        time.sleep(1)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
