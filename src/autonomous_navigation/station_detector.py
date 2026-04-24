import itertools
import math
import time


class ChargingStationDetector:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.precise_center_map = None
        self.precise_seen_count = 0
        self.last_precise_time = 0.0

        self.coarse_center_map = None
        self.last_coarse_time = 0.0

    def _polar_to_map_points(self, scan, robot_x, robot_y, robot_yaw):
        pts = []
        ang = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < min(scan.range_max, self.config.station_max_detection_range):
                lx = r * math.cos(ang)
                ly = r * math.sin(ang)
                mx = robot_x + (math.cos(robot_yaw) * lx - math.sin(robot_yaw) * ly)
                my = robot_y + (math.sin(robot_yaw) * lx + math.cos(robot_yaw) * ly)
                pts.append((mx, my))
            ang += scan.angle_increment
        return pts

    def _cluster_points(self, points):
        if not points:
            return []

        clusters = []
        current = [points[0]]
        for p in points[1:]:
            if math.hypot(p[0] - current[-1][0], p[1] - current[-1][1]) <= self.config.station_cluster_dist:
                current.append(p)
            else:
                if len(current) >= self.config.station_min_cluster_points:
                    clusters.append(current)
                current = [p]

        if len(current) >= self.config.station_min_cluster_points:
            clusters.append(current)

        centers = []
        for c in clusters:
            sx = sum(p[0] for p in c)
            sy = sum(p[1] for p in c)
            centers.append((sx / len(c), sy / len(c)))

        return centers[: self.config.station_max_pillars]

    def _is_square(self, points):
        dists = sorted(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in itertools.combinations(points, 2)
        )
        side = self.config.station_side_len
        diag = math.sqrt(2.0) * side
        tol = self.config.station_geom_tol

        if len(dists) != 6:
            return False

        return (
            abs(dists[0] - side) < tol
            and abs(dists[1] - side) < tol
            and abs(dists[2] - side) < tol
            and abs(dists[3] - side) < tol
            and abs(dists[4] - diag) < tol
            and abs(dists[5] - diag) < tol
        )

    def _compute_centroid(self, points):
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        return (sx / len(points), sy / len(points))

    def process_scan(self, scan, robot_x, robot_y, robot_yaw):
        pts = self._polar_to_map_points(scan, robot_x, robot_y, robot_yaw)
        if len(pts) < self.config.station_min_cluster_points:
            return

        pillars = self._cluster_points(pts)
        if len(pillars) < 2:
            return

        coarse = self._compute_centroid(pillars)
        if self.coarse_center_map is None:
            self.coarse_center_map = coarse
        else:
            alpha = self.config.station_coarse_ema_alpha
            self.coarse_center_map = (
                (1.0 - alpha) * self.coarse_center_map[0] + alpha * coarse[0],
                (1.0 - alpha) * self.coarse_center_map[1] + alpha * coarse[1],
            )
        self.last_coarse_time = time.time()

        if len(pillars) < 4:
            return

        max_combos = self.config.station_max_square_combinations
        checked = 0
        for combo in itertools.combinations(pillars, 4):
            if checked >= max_combos:
                break
            checked += 1

            if self._is_square(combo):
                center = self._compute_centroid(combo)
                if self.precise_center_map is None:
                    self.precise_center_map = center
                else:
                    alpha = self.config.station_precise_ema_alpha
                    self.precise_center_map = (
                        (1.0 - alpha) * self.precise_center_map[0] + alpha * center[0],
                        (1.0 - alpha) * self.precise_center_map[1] + alpha * center[1],
                    )

                self.precise_seen_count += 1
                self.last_precise_time = time.time()
                return

    def has_precise_center(self):
        return self.precise_seen_count >= self.config.station_min_precise_observations

    def get_precise_center_map(self):
        if self.has_precise_center():
            return self.precise_center_map
        return None

    def get_coarse_center_map(self):
        return self.coarse_center_map
