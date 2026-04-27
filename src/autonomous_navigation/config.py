import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

KEY_POINTS: Dict[str, Tuple[float, float]] = {
    "A": (4.280, 1.735),
    "B": (5.280, 1.735),
    "C": (4.880, 2.535),
    "D": (5.080, 5.740),
    "E": (5.880, 8.145),
    "F": (5.480, 10.545),
    "DOOR": (6.280, 11.685),
    "MIDWAY_DOOR": (7.810, 11.685),
    "Q": (9.115, 14.590),
    "U": (1.075, 16.190),
    "R": (7.310, 16.190),
    "R_BIS": (7.810, 16.190),
    "S": (3.675, 14.190),
    "T": (1.275, 14.990),
    "BASE": (3.475, 15.390),
}


@dataclass
class NavigationConfig:
    swap_xy: bool = False

    inflation_radius_m: float = 0.07
    nearest_free_search_radius_m: float = 0.45
    treat_unknown_as_free: bool = True
    planner_heuristic_weight: float = 1.00
    planner_directness_bias: float = 0.8
    planner_unknown_cell_penalty: float = 2.20
    planner_turn_penalty: float = 0.20
    planner_reverse_progress_penalty: float = 0.80
    planner_straight_path_shortcut: bool = True
    planner_simplify_path: bool = True

    max_speed: float = 0.28
    kp_linear: float = 0.5
    kp_angular: float = 1.0
    xy_tolerance: float = 0.15
    yaw_tolerance: float = 0.1
    yaw_stop_threshold: float = 0.50
    follow_lookahead_distance: float = 0.30
    min_motion_linear_speed: float = 0.04

    path_min_waypoint_spacing: float = 0.30

    safe_stop_distance: float = 0.25
    caution_distance: float = 0.28
    follow_block_trigger_distance: float = 0.30
    collision_stop_distance: float = 0.18
    turn_side_clearance: float = 0.12

    lidar_front_cone_deg: float = 90.0
    lidar_side_cone_deg: float = 40.0

    avoid_turn_speed: float = 0.6
    avoid_forward_speed: float = 0.12
    avoid_turn_tolerance: float = 0.12
    avoid_pivot_distance: float = 0.30
    avoid_pivot_distance_candidates: Tuple[float, ...] = (0.28, 0.38, 0.50, 0.65)
    avoid_pivot_angle_candidates_deg: Tuple[float, ...] = (20.0, 30.0, 45.0, 60.0, 75.0)
    avoid_goal_distance_slack: float = 0.25
    avoid_pivot_reach_tolerance: float = 0.18
    avoid_search_timeout: float = 0.8
    avoid_max_search_cycles: int = 6
    avoid_retry_cooldown: float = 1.0
    wall_follow_speed: float = 0.07

    progress_epsilon: float = 0.12
    stuck_timeout: float = 4.0
    replan_cooldown: float = 0.8

    follow_obstacle_hit_reset_s: float = 1.2
    follow_obstacle_hit_threshold: int = 3

    backup_min_front_distance: float = 0.20
    backup_recover_distance: float = 0.34
    backup_speed: float = 0.08

    door_required_y_threshold: float = 8.5
    door_forced_targets: Tuple[str, ...] = ("Q", "R", "R_BIS", "S", "T", "BASE")

    phase2_enabled: bool = True
    phase2_trigger_targets: Tuple[str, ...] = ("BASE",)
    phase2_preset_route: Tuple[str, ...] = ("R", "U", "T", "S", "Q", "R", "BASE")
    passadis_max_speed: float = 0.38

    phase3_enabled: bool = True
    phase3_search_fallback_targets: Tuple[str, ...] = ("Q", "R")
    phase3_dock_xy_tolerance: float = 0.22
    phase3_retry_cooldown: float = 1.5
    phase3_use_map_prior: bool = True
    phase3_map_prior_point_name: str = "BASE"
    phase3_max_target_distance_m: float = 25.0
    phase_marker_xy_tolerance: float = 0.35

    station_max_detection_range: float = 2.2
    station_cluster_dist: float = 0.05
    station_min_cluster_points: int = 3
    station_max_pillars: int = 20
    station_side_len: float = 0.40
    station_geom_tol: float = 0.12
    station_max_square_combinations: int = 500
    station_coarse_ema_alpha: float = 0.25
    station_precise_ema_alpha: float = 0.20
    station_min_precise_observations: int = 3
    station_min_coarse_observations: int = 4
    station_center_max_age_s: float = 6.0
    station_coarse_max_jump_m: float = 0.80

    log_scan_stride: int = 8
    log_obstacle_max_range: float = 3.0
    log_max_obstacle_points: int = 20
    log_max_station_pillars: int = 8

    door_align_tolerance: float = 0.10
    door_heading_kp: float = 1.4
    door_forward_speed: float = 0.12
    door_left_opening_distance: float = 1.2
    door_search_min_time: float = 0.7
    door_left_center_x: float = 7.30
    door_left_center_x_tolerance: float = 0.18
    door_second_cross_distance: float = 3.00

    status_print_period: float = 1.0
    marker_publish_period: float = 0.5


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euler_from_quaternion(q) -> Tuple[float, float, float]:
    t0 = +2.0 * (q.w * q.x + q.y * q.z)
    t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (q.w * q.y - q.z * q.x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class CoordinateAdapter:
    def __init__(self, swap_xy: bool = False):
        self.swap_xy = swap_xy

        self.frame_aligned = False
        self.align_cos = 1.0
        self.align_sin = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.ext_origin_x = 0.0
        self.ext_origin_y = 0.0

    def set_frame_alignment(
        self,
        ext_x: float,
        ext_y: float,
        ext_yaw: float,
        map_x: float,
        map_y: float,
        map_yaw: float,
    ):
        d_yaw = normalize_angle(map_yaw - ext_yaw)
        self.align_cos = math.cos(d_yaw)
        self.align_sin = math.sin(d_yaw)
        self.map_origin_x = map_x
        self.map_origin_y = map_y
        sx, sy = self._apply_swap(ext_x, ext_y)
        self.ext_origin_x = sx
        self.ext_origin_y = sy
        self.frame_aligned = True

    def _apply_swap(self, x: float, y: float) -> Tuple[float, float]:
        return (y, x) if self.swap_xy else (x, y)

    def to_internal_xy(self, x: float, y: float) -> Tuple[float, float]:
        sx, sy = self._apply_swap(x, y)
        if not self.frame_aligned:
            return (sx, sy)

        dx = sx - self.ext_origin_x
        dy = sy - self.ext_origin_y
        mx = self.map_origin_x + (self.align_cos * dx - self.align_sin * dy)
        my = self.map_origin_y + (self.align_sin * dx + self.align_cos * dy)
        return (mx, my)

    def to_external_xy(self, x: float, y: float) -> Tuple[float, float]:
        if self.frame_aligned:
            dx = x - self.map_origin_x
            dy = y - self.map_origin_y
            sx = self.ext_origin_x + (self.align_cos * dx + self.align_sin * dy)
            sy = self.ext_origin_y + (-self.align_sin * dx + self.align_cos * dy)
        else:
            sx, sy = x, y

        return self._apply_swap(sx, sy)

    def format_external_xy(self, xy: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if xy is None:
            return None
        ex, ey = self.to_external_xy(xy[0], xy[1])
        return (round(ex, 2), round(ey, 2))

    def external_yaw_to_internal(self, yaw_external: float) -> float:
        if not self.frame_aligned:
            return normalize_angle(yaw_external)

        align_yaw = math.atan2(self.align_sin, self.align_cos)
        return normalize_angle(yaw_external + align_yaw)

    def internal_yaw_to_external(self, yaw_internal: float) -> float:
        align_yaw = math.atan2(self.align_sin, self.align_cos) if self.frame_aligned else 0.0
        yaw_external = normalize_angle(yaw_internal - align_yaw)

        if self.swap_xy:
            yaw_external = normalize_angle((math.pi / 2.0) - yaw_external)

        return yaw_external
