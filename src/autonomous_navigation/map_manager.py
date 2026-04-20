from nav_msgs.msg import OccupancyGrid


class MapManager:
    def __init__(self, coord_adapter=None, prefer_base_map_for_planning=True, base_map_in_external_frame=True):
        self.coord_adapter = coord_adapter
        self.prefer_base_map_for_planning = prefer_base_map_for_planning
        self.base_map_in_external_frame = base_map_in_external_frame

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

        self._alignment_status_logged = False

    def active_map_source(self):
        if self.prefer_base_map_for_planning and self.base_map_received:
            return "base"
        if self.map_received:
            return "slam"
        if self.base_map_received:
            return "base"
        return "none"

    def _log_alignment_status(self, logger=None):
        if self._alignment_status_logged or logger is None:
            return
        if not (self.map_received and self.base_map_received):
            return

        if self.base_map_in_external_frame:
            logger.info(
                "Using base_map in external frame with runtime alignment from initial pose."
            )
            self._alignment_status_logged = True
            return

        if self.maps_aligned():
            logger.info("SLAM map and base_map aligned. Planner can combine both occupancies.")
        else:
            logger.warn(
                "SLAM map and base_map are NOT aligned (size/resolution/origin mismatch). "
                "Planner will rely on active map source only."
            )
        self._alignment_status_logged = True

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
        self._log_alignment_status(logger)

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
        self._log_alignment_status(logger)

    def get_active_map_info(self):
        source = self.active_map_source()
        if source == "slam":
            return self.map_width, self.map_height, self.map_resolution, self.map_origin
        if source == "base":
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
        source = self.active_map_source()

        if source == "slam":
            width = self.map_width
            height = self.map_height
            resolution = self.map_resolution
            origin = self.map_origin
            if width <= 0 or height <= 0:
                return None
            gx = int((x - origin[0]) / resolution)
            gy = int((y - origin[1]) / resolution)
            return (gx, gy)

        if source == "base":
            width = self.base_map_width
            height = self.base_map_height
            resolution = self.base_map_resolution
            origin = self.base_map_origin
            if width <= 0 or height <= 0:
                return None

            bx, by = x, y
            if self.base_map_in_external_frame and self.coord_adapter is not None:
                bx, by = self.coord_adapter.to_external_xy(x, y)

            gx = int((bx - origin[0]) / resolution)
            gy = int((by - origin[1]) / resolution)
            return (gx, gy)

        return None

    def in_bounds(self, gx, gy):
        source = self.active_map_source()
        if source == "base":
            width = self.base_map_width
            height = self.base_map_height
        else:
            width = self.map_width
            height = self.map_height
        return 0 <= gx < width and 0 <= gy < height

    def clamp_to_map(self, gx, gy):
        source = self.active_map_source()
        if source == "base":
            width = self.base_map_width
            height = self.base_map_height
        else:
            width = self.map_width
            height = self.map_height
        if width <= 0 or height <= 0:
            return (gx, gy)
        gx = min(max(gx, 0), width - 1)
        gy = min(max(gy, 0), height - 1)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        source = self.active_map_source()

        if source == "base":
            resolution = self.base_map_resolution
            origin = self.base_map_origin
            bx = (gx * resolution) + origin[0] + (resolution / 2)
            by = (gy * resolution) + origin[1] + (resolution / 2)

            if self.base_map_in_external_frame and self.coord_adapter is not None:
                return self.coord_adapter.to_internal_xy(bx, by)
            return (bx, by)

        resolution = self.map_resolution
        origin = self.map_origin
        wx = (gx * resolution) + origin[0] + (resolution / 2)
        wy = (gy * resolution) + origin[1] + (resolution / 2)
        return (wx, wy)

    def get_cell_occupancy(self, gx, gy):
        source = self.active_map_source()
        active_val = None
        other_val = None

        if source == "base":
            if self.base_map_received and 0 <= gx < self.base_map_width and 0 <= gy < self.base_map_height:
                idx = gy * self.base_map_width + gx
                active_val = self.base_map_data[idx]

            if self.maps_aligned() and self.map_received and 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                idx = gy * self.map_width + gx
                other_val = self.map_data[idx]

        elif source == "slam":
            if self.map_received and 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                idx = gy * self.map_width + gx
                active_val = self.map_data[idx]

            if self.maps_aligned() and self.base_map_received and 0 <= gx < self.base_map_width and 0 <= gy < self.base_map_height:
                idx = gy * self.base_map_width + gx
                other_val = self.base_map_data[idx]

        vals = [v for v in [active_val, other_val] if v is not None and v >= 0]
        if not vals:
            return -1
        return max(vals)
