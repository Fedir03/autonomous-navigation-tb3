from nav_msgs.msg import OccupancyGrid


class MapManager:
    def __init__(self):
        # Dynamic map (/map)
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        self.map_received = False

        # Static/base map (/base_map)
        self.base_map_data = None
        self.base_map_resolution = 0.05
        self.base_map_origin = [0.0, 0.0]
        self.base_map_width = 0
        self.base_map_height = 0
        self.base_map_received = False

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

    def base_map_callback(self, msg: OccupancyGrid, logger=None):
        self.base_map_data = msg.data
        self.base_map_resolution = msg.info.resolution
        self.base_map_width = msg.info.width
        self.base_map_height = msg.info.height
        self.base_map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        if not self.base_map_received:
            if logger is not None:
                logger.info(
                    f"Base map received: {self.base_map_width}x{self.base_map_height} @ {self.base_map_resolution}m/px"
                )
            self.base_map_received = True

    def get_active_map_info(self):
        if self.map_received:
            return self.map_width, self.map_height, self.map_resolution, self.map_origin
        if self.base_map_received:
            return self.base_map_width, self.base_map_height, self.base_map_resolution, self.base_map_origin
        return 0, 0, 0.05, [0.0, 0.0]

    def maps_aligned(self):
        if not (self.map_received and self.base_map_received):
            return False
        return (
            self.map_width == self.base_map_width
            and self.map_height == self.base_map_height
            and abs(self.map_resolution - self.base_map_resolution) < 1e-6
            and abs(self.map_origin[0] - self.base_map_origin[0]) < 1e-6
            and abs(self.map_origin[1] - self.base_map_origin[1]) < 1e-6
        )

    def world_to_grid(self, x, y):
        width, height, resolution, origin = self.get_active_map_info()
        if width <= 0 or height <= 0:
            return None
        gx = int((x - origin[0]) / resolution)
        gy = int((y - origin[1]) / resolution)
        return (gx, gy)

    def in_bounds(self, gx, gy):
        width, height, _, _ = self.get_active_map_info()
        return 0 <= gx < width and 0 <= gy < height

    def clamp_to_map(self, gx, gy):
        width, height, _, _ = self.get_active_map_info()
        if width <= 0 or height <= 0:
            return (gx, gy)
        gx = min(max(gx, 0), width - 1)
        gy = min(max(gy, 0), height - 1)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        _, _, resolution, origin = self.get_active_map_info()
        wx = (gx * resolution) + origin[0] + (resolution / 2)
        wy = (gy * resolution) + origin[1] + (resolution / 2)
        return (wx, wy)

    def get_cell_occupancy(self, gx, gy):
        dyn = None
        base = None

        if self.map_received and 0 <= gx < self.map_width and 0 <= gy < self.map_height:
            idx_dyn = gy * self.map_width + gx
            dyn = self.map_data[idx_dyn]

        if self.base_map_received:
            if self.map_received:
                if self.maps_aligned() and 0 <= gx < self.base_map_width and 0 <= gy < self.base_map_height:
                    idx_base = gy * self.base_map_width + gx
                    base = self.base_map_data[idx_base]
            else:
                if 0 <= gx < self.base_map_width and 0 <= gy < self.base_map_height:
                    idx_base = gy * self.base_map_width + gx
                    base = self.base_map_data[idx_base]

        vals = [v for v in [dyn, base] if v is not None and v >= 0]
        if not vals:
            return -1
        return max(vals)
