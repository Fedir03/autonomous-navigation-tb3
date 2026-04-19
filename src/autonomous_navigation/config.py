import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Key reference points (external/user convention, do NOT swap)
KEY_POINTS: Dict[str, Tuple[float, float]] = {
    "A": (2.52, 1.35),
    "B": (3.72, 2.55),
    "C": (1.32, 0.95),
    "D": (3.32, 0.95),
    "DOOR": (5.92, 8.12),
    "BASE": (5.00, 11.69),
    "O": (5.10, 12.61),
    "P": (0.30, 11.01),
    "Q": (1.90, 12.21),
    "R": (7.12, 12.61),
}


@dataclass
class NavigationConfig:
    swap_xy: bool = False

    inflation_radius: int = 4

    max_speed: float = 0.2
    kp_linear: float = 0.5
    kp_angular: float = 1.0
    xy_tolerance: float = 0.15
    yaw_tolerance: float = 0.1

    safe_stop_distance: float = 0.25
    caution_distance: float = 0.40
    follow_block_trigger_distance: float = 0.28

    wall_follow_distance: float = 0.45
    avoid_turn_speed: float = 0.6
    wall_follow_speed: float = 0.08
    wall_follow_kp: float = 1.2
    wall_follow_min_time: float = 1.8

    progress_epsilon: float = 0.12
    stuck_timeout: float = 6.0
    replan_cooldown: float = 0.8

    follow_obstacle_hit_reset_s: float = 1.2
    follow_obstacle_hit_threshold: int = 3

    # Emergency reverse: if an obstacle is closer than this, back up until clear.
    backup_min_front_distance: float = 0.20
    backup_speed: float = 0.08

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

        # Transform from external/user frame -> SLAM map frame.
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
        # Rotate/translate external coordinates so the provided initial pose matches current SLAM pose.
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
