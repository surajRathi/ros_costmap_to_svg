#! /usr/bin/python3
import base64
import io
import pathlib
from typing import Optional, Tuple, Callable

import cv2
import numpy as np
import rospy
import yaml
from PIL import Image
from geometry_msgs.msg import Pose
from map_to_svg.srv import FinishEditing, FinishEditingRequest, FinishEditingResponse
from map_to_svg.srv import ListMaps, ListMapsRequest, ListMapsResponse
from map_to_svg.srv import SetMap, SetMapRequest, SetMapResponse
from map_to_svg.srv import StartEditing, StartEditingRequest, StartEditingResponse
from nav_msgs.msg import MapMetaData, OccupancyGrid
from nav_msgs.srv import GetMap, GetMapRequest, GetMapResponse
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from lxml import etree

"""
Contents of a map directory:
- map.yaml [REQUIRED]
- map.pgm [REQUIRED]
- map.svg
- obs.pgm (generated from map.svg via the frontend)
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


# TODO: Support PNG maps!
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
            width, height = map(int, dims.decode('ASCII').strip().split(' '))
            maximum_value = int(max_val.decode('ASCII').strip())
            if maximum_value > 255:
                rospy.logerr(f"No support for reading pgm files with max value greater than {maximum_value}")
                return None

            map_data = f.read()
            if len(map_data) != width * height:
                rospy.logerr(f"Invalid PGM file read {len(map_data)}, expected {width * height}.")
                return None

            arr: np.ndarray = (np.frombuffer(map_data, dtype=np.uint8)).reshape(width, height)
            return arr

        except None:
            rospy.logerr("PGM Read Error")
            return None


class MapPublisher:

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, new_name: Optional[str]):
        self.loaded = False
        self._name = None
        if new_name is not None:
            if self.load(new_name):
                self.loaded = True
                self._name = new_name
                rospy.loginfo(f'Loaded {new_name}.')
            else:
                rospy.logerr(f'Could not load map {new_name}.')

        if not self.loaded:
            # Doesn't really need to be done, but good for bug checking.
            self.data = None
            self.meta_data = None
            self.map_data = None
            self.obs_map_data = None
            self.svg_map_data = None

    def __init__(self, frame_id: str, data_dir: str, name: Optional[str] = None):
        self.temp = None
        self.loaded = False
        self.data: Optional[np.ndarray] = None
        self.meta_data: Optional[MapMetaData] = None
        self.map_data: Optional[numpy_msg(OccupancyGrid)] = None
        self.obs_map_data: Optional[numpy_msg(OccupancyGrid)] = None
        self.svg_map_data: Optional[String] = None

        self._name: Optional[str] = None

        self.frame_id = frame_id
        self.data_dir = data_dir

        # ROS Handles
        self.meta_pub = rospy.Publisher('map_metadata', data_class=MapMetaData, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('map', data_class=numpy_msg(OccupancyGrid), queue_size=1, latch=True)
        self.obs_map_pub = rospy.Publisher('obs_map', data_class=numpy_msg(OccupancyGrid), queue_size=1, latch=True)
        self.svg_map_pub = rospy.Publisher('svg_map', String, queue_size=1, latch=True)

        self.static_map_srv = rospy.Service('static_map', GetMap, self.get_map)
        self.list_maps_srv = rospy.Service('map_server/list_maps', ListMaps, self.list_maps)
        self.set_map_srv = rospy.Service('map_server/set_map', SetMap, self.set_map)
        self.start_editing_srv = rospy.Service('map_server/start_editing', StartEditing, self.start_editing)
        self.finish_editing_srv = rospy.Service('map_server/finish_editing', FinishEditing, self.finish_editing)

        # Initialize self
        self.name = name

    def get_map(self, req: GetMapRequest) -> Optional[GetMapResponse]:
        # TODO: Properly deal with failure in other services
        if not self.loaded:
            return None

        resp = GetMapResponse()
        resp.map.header = self.map_data.header
        resp.map.info = self.map_data.info
        resp.map.data = self.map_data.data.astype(np.int8).tolist()
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
        if req.svg_data == '':
            return FinishEditingResponse(success=1)

        # Save the svg without the
        with (pathlib.Path(self.data_dir) / self.name / 'map.svg').open('w') as f:
            f.write(req.svg_data)

        # TODO: Render the obstacle svg directly onto an occupancy grid
        # Save the obstacle png as a pgm
        # The below line is correct.
        # noinspection PyTypeChecker
        im = np.array(Image.open(io.BytesIO(base64.b64decode(req.png_data[req.png_data.find(','):]))))
        # 0: Obstacle
        # 254: free space
        # 204: unknowns
        im1 = np.zeros(im.shape[:2], dtype=np.uint8) + 204
        im1[np.any(im[:, :, :3] != 0, axis=-1)] = 0

        with (pathlib.Path(self.data_dir) / self.name / 'obs.pgm').open('wb') as f:
            f.write(b'P5\n')
            f.write(f"# CREATOR: map_server.py {self.meta_data.resolution:.3f} m/pix\n".encode())
            f.write(f"{self.meta_data.width} {self.meta_data.height}\n".encode())
            f.write(b"255\n")
            f.write(im1.tobytes())

        # Reload
        self.load(self.name)

        return FinishEditingResponse(success=(1 if self.loaded else 0))

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

        obs_pgm_path = pathlib.Path(self.data_dir) / name / 'obs.pgm'
        if obs_pgm_path.exists():
            map_arr = read_pgm(obs_pgm_path)
            if map_arr is None:
                rospy.logerr("Could not load the obs pgm file.")
                return False

            width, height = map_arr.shape
            if width != self.meta_data.width or height != self.meta_data.height:
                rospy.logerr('obs pgm is the wrong shape')
            else:
                self.obs_map_data: numpy_msg(OccupancyGrid) = numpy_msg(OccupancyGrid)()
                self.obs_map_data.header.frame_id = self.frame_id
                self.obs_map_data.header.stamp = self.meta_data.map_load_time
                self.obs_map_data.info = self.meta_data
                self.obs_map_data.data = value_interpreter(map_arr.reshape(-1)).astype(np.uint8)

                self.obs_map_pub.publish(self.obs_map_data)
                rospy.loginfo('Publishing to the topic obs_map.')

        svg_file = pathlib.Path(self.data_dir) / name / 'map.svg'
        if not svg_file.exists():
            self.create_svg(name)
        self.svg_map_data = String()

        # TODO: Horrible, horrible hack
        svg_data = svg_file.open('r').read()
        svg_data_in = svg_data.find('>') + 1
        svg_data_in1 = svg_data.find('<svg ') + len('<svg ')
        css_data = (pathlib.Path(self.data_dir) / 'map.css').open('r').read()

        self.svg_map_data.data = svg_data[:svg_data_in1] + f"width='{self.meta_data.width}' " \
                                 + svg_data[svg_data_in1:svg_data_in] \
                                 + f"<style>" + css_data + '</style>' \
                                 + svg_data[svg_data_in:]

        self.svg_map_pub.publish(self.svg_map_data)
        rospy.loginfo('Publishing to the topic svg_map.')

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

        with io.BytesIO() as f:
            Image.fromarray(image).save(f, 'PNG')
            f.seek(0)
            return base64.b64encode(f.read()).decode('ASCII')

    def create_svg(self, name=None):
        if name is None and not self.loaded:
            return ''

        if name is None:
            name = self.name
        scale = max(1, self.map_data.info.resolution / 0.01)
        img = self.map_data.data.reshape(self.meta_data.height, self.meta_data.width)

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

        with (pathlib.Path(self.data_dir) / name / 'map.svg').open('w') as f:
            f.write(
                f"<svg xmlns='http://www.w3.org/2000/svg' preserveAspectRatio='xMinYMin meet' viewBox='0 0 {img.shape[1]} {img.shape[0]}'>\n")

            f.write(f"<rect x='0' y='0' class='bg' width='{img.shape[1]}' height='{img.shape[0]}' />\n")

            for (l,) in lines:
                f.write(f"<line x1='{l[0]}' y1='{l[1]}' x2='{l[2]}' y2='{l[3]}' class='item obstacle' />\n")

            f.write("</svg>")


def main():
    rospy.init_node("map_server")

    data_dir: str = '/home/suraj/ws/src/rosjs/map_to_svg/data/final'
    frame_id: str = rospy.get_param('~frame_id', 'map')

    m = MapPublisher(frame_id, data_dir, 'office2')

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
