from nav_msgs.msg import OccupancyGrid


class MapManager:
    def __init__(self, coord_adapter=None):
        self.coord_adapter = coord_adapter

        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        self.map_received = False

    def active_map_source(self):
        if self.map_received:
            return "slam"
        return "none"

    def map_callback(self, msg: OccupancyGrid, logger=None):
        self.map_data = msg.data
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        if not self.map_received:
            if logger is not None:
                logger.info(
                    f"SLAM map received: {self.map_width}x{self.map_height} @ {self.map_resolution}m/px"
                )
            self.map_received = True

    def get_active_map_info(self):
        if self.map_received:
            return self.map_width, self.map_height, self.map_resolution, self.map_origin
        return 0, 0, 0.05, [0.0, 0.0]

    def world_to_grid(self, x, y):
        width = self.map_width
        height = self.map_height
        resolution = self.map_resolution
        origin = self.map_origin
        if width <= 0 or height <= 0:
            return None
        gx = int((x - origin[0]) / resolution)
        gy = int((y - origin[1]) / resolution)
        return (gx, gy)

    def in_bounds(self, gx, gy):
        width = self.map_width
        height = self.map_height
        return 0 <= gx < width and 0 <= gy < height

    def clamp_to_map(self, gx, gy):
        width = self.map_width
        height = self.map_height
        if width <= 0 or height <= 0:
            return (gx, gy)
        gx = min(max(gx, 0), width - 1)
        gy = min(max(gy, 0), height - 1)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        resolution = self.map_resolution
        origin = self.map_origin
        wx = (gx * resolution) + origin[0] + (resolution / 2)
        wy = (gy * resolution) + origin[1] + (resolution / 2)
        return (wx, wy)

    def get_cell_occupancy(self, gx, gy):
        if not (self.map_received and 0 <= gx < self.map_width and 0 <= gy < self.map_height):
            return -1
        idx = gy * self.map_width + gx
        return self.map_data[idx]
