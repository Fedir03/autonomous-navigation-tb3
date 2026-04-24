import math
import os
import select
import sys
import termios
import threading
import time
import tty
from copy import deepcopy

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray

from .config import CoordinateAdapter, KEY_POINTS, NavigationConfig
from .global_planner import GlobalPlanner
from .local_planner import LocalPlanner
from .map_manager import MapManager
from .pose_estimator import PoseEstimator
from .route_manager import RouteManager
from .station_detector import ChargingStationDetector
from .telemetry import Telemetry


class AutonomousNavigationNode(Node):
    def __init__(self):
        super().__init__("autonomous_navigation_node")

        self.config = NavigationConfig()
        self.coords = CoordinateAdapter(self.config.swap_xy)

        self.map_manager = MapManager(
            coord_adapter=self.coords,
            prefer_base_map_for_planning=self.config.prefer_base_map_for_planning,
        )
        self.pose_estimator = PoseEstimator(self)
        self.global_planner = GlobalPlanner(
            self.map_manager,
            self.config.inflation_radius_m,
            self.config.nearest_free_search_radius_m,
            self.config.treat_unknown_as_free,
            self.config.planner_heuristic_weight,
            self.config.planner_directness_bias,
            self.config.planner_unknown_cell_penalty,
            self.config.planner_turn_penalty,
            self.config.planner_reverse_progress_penalty,
            self.config.planner_straight_path_shortcut,
            self.config.planner_simplify_path,
            self.config.path_min_waypoint_spacing,
        )
        self.route_manager = RouteManager(self.global_planner, self.config, self.get_logger())
        self.local_planner = LocalPlanner(self.config, self.get_logger(), self.coords)
        self.telemetry = Telemetry(self, self.coords, KEY_POINTS)
        self.station_detector = ChargingStationDetector(self.config, self.get_logger())

        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        qos_map_fallback = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, "/nav_debug_markers", 10)
        self.path_pub = self.create_publisher(Path, "/nav_planned_path", 10)
        self.base_map_aligned_pub = self.create_publisher(OccupancyGrid, "/base_map_aligned", qos_map)

        self.create_subscription(Odometry, "/odom", self.pose_estimator.odom_callback, qos_best_effort)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_best_effort)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_map)
        self.create_subscription(OccupancyGrid, "/base_map", self.base_map_callback, qos_map)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_map_fallback)
        self.create_subscription(OccupancyGrid, "/base_map", self.base_map_callback, qos_map_fallback)

        self.awaiting_user_input = False
        self.status_print_requests = 0
        self.status_lock = threading.Lock()
        self.pending_frame_alignment = None
        self.frame_alignment_done = False
        self.last_marker_publish_time = 0.0
        self.latest_base_map_msg = None
        self.base_map_origin_override_internal = None

        self.current_phase = "I"
        self.phase_entry_y = KEY_POINTS["DOOR"][1] + 0.20
        self.phase_exit_y = KEY_POINTS["DOOR"][1] - 0.20
        self.last_runtime_log_time = 0.0
        self.runtime_log_path = None
        self.runtime_log_file = None
        self._open_runtime_log_file()

        self.latest_obstacle_points_map = []
        self.phase2_active = False
        self.phase2_entered_passadis = False
        self.phase2_returned_base = False
        self.phase3_arm_after_phase2 = False
        self.phase3_started = False
        self.phase2_start_time = 0.0
        self.phase2_max_duration_s = 180.0
        self.phase2_repeat_count = 0
        self.phase2_max_repeats = 3

        self.phase3_pending = False
        self.phase3_active = False
        self.phase3_docking_finished = False
        self.phase3_last_start_attempt = 0.0
        self.phase3_refine_attempted = False
        self.phase3_started = False

        self.timer = self.create_timer(0.1, self.control_loop)
        threading.Thread(target=self.input_thread, daemon=True).start()
        threading.Thread(target=self.spacebar_status_thread, daemon=True).start()

    def _open_runtime_log_file(self):
        logs_dir = os.path.join("/tmp", "ros_autonomous_navigation_logs")
        os.makedirs(logs_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.runtime_log_path = os.path.join(logs_dir, f"navigation_runtime_{stamp}.csv")
        self.runtime_log_file = open(self.runtime_log_path, "w", encoding="utf-8", buffering=1)
        self.runtime_log_file.write(
            "timestamp,current_phase,robot_x,robot_y,robot_yaw,obstacle_points,station_pillars\n"
        )
        print(f"[LOG] Runtime log: {self.runtime_log_path}")
        self.get_logger().info(f"Runtime log enabled: {self.runtime_log_path}")

    def close_runtime_log_file(self):
        if self.runtime_log_file is not None:
            self.runtime_log_file.close()
            self.runtime_log_file = None

    def _update_phase_from_external_y(self, ext_y):
        if self.current_phase == "I" and ext_y >= self.phase_entry_y:
            self.current_phase = "II"
        elif self.current_phase == "II" and ext_y <= self.phase_exit_y:
            self.current_phase = "I"

    def _format_points_compact(self, points_map, max_points):
        if not points_map:
            return ""

        payload = []
        for p in points_map[:max_points]:
            ex, ey = self.coords.to_external_xy(p[0], p[1])
            payload.append("{:.2f},{:.2f}".format(ex, ey))
        return ";".join(payload)

    def _extract_obstacle_points_map(self, scan):
        if self.pose_estimator.pose_source == "none" and not self.pose_estimator.initial_pose_received:
            return []

        points = []
        ang = scan.angle_min
        stride = max(1, int(self.config.log_scan_stride))
        max_r = min(scan.range_max, self.config.log_obstacle_max_range)
        min_passadis_y = KEY_POINTS["DOOR"][1] - 0.10

        for i, r in enumerate(scan.ranges):
            if (i % stride) != 0:
                ang += scan.angle_increment
                continue

            if scan.range_min < r < max_r:
                lx = r * math.cos(ang)
                ly = r * math.sin(ang)
                mx = self.pose_estimator.current_x + (
                    math.cos(self.pose_estimator.current_yaw) * lx - math.sin(self.pose_estimator.current_yaw) * ly
                )
                my = self.pose_estimator.current_y + (
                    math.sin(self.pose_estimator.current_yaw) * lx + math.cos(self.pose_estimator.current_yaw) * ly
                )
                ex, ey = self.coords.to_external_xy(mx, my)
                if ey >= min_passadis_y:
                    points.append((mx, my))

            ang += scan.angle_increment

        return points[: self.config.log_max_obstacle_points]

    def _update_phase_from_mission_state(self):
        ex, ey = self.coords.to_external_xy(
            self.pose_estimator.current_x,
            self.pose_estimator.current_y,
        )

        if self.phase2_active and (not self.phase2_entered_passadis):
            r_bis = KEY_POINTS.get("R_BIS")
            if r_bis is not None:
                if math.hypot(ex - r_bis[0], ey - r_bis[1]) <= self.config.phase_marker_xy_tolerance:
                    self.phase2_entered_passadis = True

        if self.phase2_active and self.phase2_entered_passadis and (not self.phase2_returned_base):
            base = KEY_POINTS["BASE"]
            now = time.time()
            phase2_timeout = (now - self.phase2_start_time) > self.phase2_max_duration_s
            at_base = math.hypot(ex - base[0], ey - base[1]) <= self.config.phase_marker_xy_tolerance
            
            if at_base or phase2_timeout:
                station_detected = self.station_detector.get_first_detected_precise_center() is not None
                
                if phase2_timeout:
                    print(f"[Phase 2] Timeout after {self.phase2_max_duration_s}s.")
                
                if station_detected or self.phase2_repeat_count >= self.phase2_max_repeats:
                    if not station_detected:
                        print(f"[Phase 2] Max repeats ({self.phase2_max_repeats}) reached. Proceeding without station detection.")
                    self.phase2_returned_base = True
                    self.phase2_active = False
                    if self.phase3_arm_after_phase2:
                        self.phase3_pending = True
                        self.phase3_arm_after_phase2 = False
                else:
                    print(f"[Phase 2] No station detected. Repeating exploration (attempt {self.phase2_repeat_count + 1}/{self.phase2_max_repeats}).")
                    self.phase2_repeat_count += 1
                    self.phase2_entered_passadis = False
                    self.phase2_start_time = now

        if self.phase3_started or self.phase3_active or self.phase3_pending:
            self.current_phase = "III"
        elif self.phase2_active and self.phase2_entered_passadis and (not self.phase2_returned_base):
            self.current_phase = "II"
        else:
            self.current_phase = "I"

    def write_runtime_log(self, now):
        if self.runtime_log_file is None:
            return
        if (now - self.last_runtime_log_time) < 1.0:
            return

        ex, ey = self.coords.to_external_xy(self.pose_estimator.current_x, self.pose_estimator.current_y)
        eyaw = self.coords.internal_yaw_to_external(self.pose_estimator.current_yaw)
        self._update_phase_from_mission_state()

        obstacle_points = self._format_points_compact(
            self.latest_obstacle_points_map,
            self.config.log_max_obstacle_points,
        )
        station_pillars = self._format_points_compact(
            self.station_detector.get_recent_pillars_map(),
            self.config.log_max_station_pillars,
        )

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.runtime_log_file.write(
            f"{ts},{self.current_phase},{ex:.3f},{ey:.3f},{eyaw:.4f},\"{obstacle_points}\",\"{station_pillars}\"\n"
        )
        self.last_runtime_log_time = now

    def map_callback(self, msg: OccupancyGrid):
        self.map_manager.map_callback(msg, self.get_logger())

    def scan_callback(self, msg: LaserScan):
        self.local_planner.scan_callback(msg)

        if self.pose_estimator.pose_source == "none" and not self.pose_estimator.initial_pose_received:
            return

        self.station_detector.process_scan(
            msg,
            self.pose_estimator.current_x,
            self.pose_estimator.current_y,
            self.pose_estimator.current_yaw,
        )
        self.latest_obstacle_points_map = self._extract_obstacle_points_map(msg)

    def _reset_phase3_state(self):
        self.phase3_pending = False
        self.phase3_active = False
        self.phase3_docking_finished = False
        self.phase3_last_start_attempt = 0.0
        self.phase3_refine_attempted = False
        self.phase3_started = False

    def _start_phase3_segment(self, now: float, target_internal, label: str) -> bool:
        current_xy = (self.pose_estimator.current_x, self.pose_estimator.current_y)
        self.route_manager.set_route(
            target_internal,
            [target_internal],
            now,
            door_transition_required=False,
            door_waypoint=None,
            speed_mode="normal",
        )
        self.local_planner.reset_for_new_route()

        ex, ey = self.coords.to_external_xy(target_internal[0], target_internal[1])
        print(
            "Phase 3: planning toward {} at external ({:.2f}, {:.2f})".format(
                label,
                ex,
                ey,
            )
        )

        ok = self.route_manager.start_next_segment(current_xy, now)
        self.phase3_last_start_attempt = now
        if not ok:
            print("Phase 3: could not start segment, will retry.")
        else:
            self.phase3_started = True
        return ok

    def _phase3_target_is_valid(self, current_xy, target_xy):
        if target_xy is None:
            return False

        d = math.hypot(target_xy[0] - current_xy[0], target_xy[1] - current_xy[1])
        if d > self.config.phase3_max_target_distance_m:
            return False

        g = self.map_manager.world_to_grid(target_xy[0], target_xy[1])
        if g is None:
            return False

        if not self.map_manager.in_bounds(g[0], g[1]):
            return False

        return True

    def _phase3_get_map_prior_center(self):
        if not self.config.phase3_use_map_prior:
            return None

        name = self.config.phase3_map_prior_point_name
        if name not in KEY_POINTS:
            return None

        p = KEY_POINTS[name]
        return self.coords.to_internal_xy(p[0], p[1])

    def _maybe_start_phase3(self, now: float):
        if not self.config.phase3_enabled:
            return
        if self.phase3_docking_finished:
            return
        if not (self.phase3_pending or self.phase3_active):
            return
        if self.route_manager.is_moving:
            return
        if now - self.phase3_last_start_attempt < self.config.phase3_retry_cooldown:
            return

        current_xy = (self.pose_estimator.current_x, self.pose_estimator.current_y)
        
        first_detected = self.station_detector.get_first_detected_precise_center()
        if (first_detected is not None) and self._phase3_target_is_valid(current_xy, first_detected):
            d = math.hypot(first_detected[0] - current_xy[0], first_detected[1] - current_xy[1])
            if d <= self.config.phase3_dock_xy_tolerance:
                print("Phase 3 docking complete: robot is at charging-station center.")
                self.phase3_pending = False
                self.phase3_active = False
                self.phase3_docking_finished = True
                return

            if self._start_phase3_segment(now, first_detected, "detected station center"):
                self.phase3_pending = False
                self.phase3_active = True
            return
        
        precise = self.station_detector.get_precise_center_map()
        if (precise is not None) and self._phase3_target_is_valid(current_xy, precise):
            d = math.hypot(precise[0] - current_xy[0], precise[1] - current_xy[1])
            if d <= self.config.phase3_dock_xy_tolerance:
                print("Phase 3 docking complete: robot is at charging-station center.")
                self.phase3_pending = False
                self.phase3_active = False
                self.phase3_docking_finished = True
                return

            if self._start_phase3_segment(now, precise, "current station center"):
                self.phase3_pending = False
                self.phase3_active = True
            return

        coarse = self.station_detector.get_coarse_center_map()
        if (coarse is not None) and self._phase3_target_is_valid(current_xy, coarse):
            d_coarse = math.hypot(coarse[0] - current_xy[0], coarse[1] - current_xy[1])
            if d_coarse > self.config.phase3_dock_xy_tolerance:
                if self._start_phase3_segment(now, coarse, "station coarse estimate"):
                    self.phase3_pending = False
                    self.phase3_active = True
                    self.phase3_refine_attempted = True
            else:
                print("Phase 3: at coarse station area, waiting for precise center refinement.")
                self.phase3_last_start_attempt = now
            return

        print("Phase 3: waiting for station detection before docking.")
        self.phase3_last_start_attempt = now

    def base_map_callback(self, msg: OccupancyGrid):
        self.latest_base_map_msg = msg
        aligned = self._build_aligned_base_map_msg(msg)
        self.base_map_aligned_pub.publish(aligned)
        self.map_manager.base_map_callback(aligned, self.get_logger())

    def _build_aligned_base_map_msg(self, msg: OccupancyGrid) -> OccupancyGrid:
        aligned = deepcopy(msg)
        if self.base_map_origin_override_internal is not None:
            aligned.info.origin.position.x = self.base_map_origin_override_internal[0]
            aligned.info.origin.position.y = self.base_map_origin_override_internal[1]
        return aligned

    def _update_base_map_alignment_from_initial_pose(self):
        if not self.config.base_map_dynamic_alignment_enabled:
            return
        if not self.frame_alignment_done:
            return

        ref_init = self.config.base_map_reference_initial_external_xy
        ref_origin = self.config.base_map_reference_origin_map_xy
        anchor_external_x = ref_init[0] + ref_origin[0]
        anchor_external_y = ref_init[1] + ref_origin[1]
        ox, oy = self.coords.to_internal_xy(anchor_external_x, anchor_external_y)

        self.base_map_origin_override_internal = [ox, oy]
        self.map_manager.set_base_map_origin_override(self.base_map_origin_override_internal)
        self.get_logger().info(
            f"Applied dynamic base_map origin from initial pose: ({ox:.3f}, {oy:.3f})"
        )

        if self.latest_base_map_msg is not None:
            aligned = self._build_aligned_base_map_msg(self.latest_base_map_msg)
            self.base_map_aligned_pub.publish(aligned)
            self.map_manager.base_map_callback(aligned, self.get_logger())

    def try_align_frames(self):
        if self.frame_alignment_done or self.pending_frame_alignment is None:
            return False

        if not self.pose_estimator.get_robot_pose():
            return False

        init_x, init_y, yaw_rad = self.pending_frame_alignment
        self.coords.set_frame_alignment(
            init_x,
            init_y,
            yaw_rad,
            self.pose_estimator.current_x,
            self.pose_estimator.current_y,
            self.pose_estimator.current_yaw,
        )
        self.frame_alignment_done = True
        self._update_base_map_alignment_from_initial_pose()
        self.get_logger().info("Coordinate frames aligned using TF map pose.")
        return True

    def read_user_line(self, prompt: str) -> str:
        self.awaiting_user_input = True
        try:
            return input(prompt)
        finally:
            self.awaiting_user_input = False

    def parse_objective_token(self, token):
        token = token.strip()
        if not token:
            return None

        token_upper = token.upper()
        if token_upper in KEY_POINTS:
            return {
                "external": KEY_POINTS[token_upper],
                "name": token_upper,
                "label": token_upper,
            }

        coord_text = token.strip().strip("()[]")
        parts = [p.strip() for p in coord_text.split(",")]
        if len(parts) != 2:
            return None

        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            return None

        return {
            "external": (x, y),
            "name": None,
            "label": "{:.2f},{:.2f}".format(x, y),
        }

    def parse_objective_sequence(self, user_input):
        tokens = [t.strip() for t in user_input.split(";")]
        if any(t == "" for t in tokens):
            return None

        objectives = []
        for token in tokens:
            objective = self.parse_objective_token(token)
            if objective is None:
                return None
            objectives.append(objective)

        return objectives

    def _append_external_waypoint(self, route_external, ext_wp):
        int_wp = self.coords.to_internal_xy(ext_wp[0], ext_wp[1])
        if route_external:
            prev = route_external[-1]
            prev_int = self.coords.to_internal_xy(prev[0], prev[1])
            if math.hypot(int_wp[0] - prev_int[0], int_wp[1] - prev_int[1]) < 0.05:
                return
        route_external.append(ext_wp)

    def generate_passadis_preset_waypoints(self):
        waypoints = []
        for name in self.config.phase2_preset_route:
            if name in KEY_POINTS:
                self._append_external_waypoint(waypoints, KEY_POINTS[name])
        return waypoints

    def generate_door_chain_waypoints(self):
        waypoints = []
        for name in ("DOOR", "MIDWAY_DOOR", "R_BIS"):
            if name in KEY_POINTS:
                self._append_external_waypoint(waypoints, KEY_POINTS[name])
        return waypoints

    def build_mandatory_route(self, objectives, current_external_xy):
        if not objectives:
            return [], [], False, None, False

        door_external = KEY_POINTS["DOOR"]
        current_ext_y = current_external_xy[1]

        door_pass_threshold = max(self.config.door_required_y_threshold, door_external[1] - 0.20)
        door_passed = current_ext_y > door_pass_threshold
        route_external = []
        phase2_injected = False

        for objective in objectives:
            ext_wp = objective["external"]
            obj_name = objective["name"]

            is_door_obj = (obj_name == "DOOR") or (
                math.hypot(ext_wp[0] - door_external[0], ext_wp[1] - door_external[1]) < 0.30
            )
            forced_by_name = (
                obj_name in self.config.door_forced_targets
                if obj_name is not None
                else False
            )

            needs_door_first = (not door_passed) and (not is_door_obj) and (
                forced_by_name
            )

            if needs_door_first:
                for door_wp in self.generate_door_chain_waypoints():
                    self._append_external_waypoint(route_external, door_wp)
                door_passed = True

            phase2_target = (
                obj_name in self.config.phase2_trigger_targets
                if obj_name is not None
                else False
            )
            if (
                self.config.phase2_enabled
                and phase2_target
                and door_passed
                and (not phase2_injected)
                and (not is_door_obj)
            ):
                for preset_wp in self.generate_passadis_preset_waypoints():
                    self._append_external_waypoint(route_external, preset_wp)
                phase2_injected = True

            self._append_external_waypoint(route_external, ext_wp)

            if is_door_obj:
                door_passed = True

        mandatory_internal = [
            self.coords.to_internal_xy(p[0], p[1])
            for p in route_external
        ]

        return mandatory_internal, route_external, False, None, phase2_injected

    def queue_status_print(self):
        with self.status_lock:
            self.status_print_requests += 1

    def consume_status_print_requests(self) -> int:
        with self.status_lock:
            requests = self.status_print_requests
            self.status_print_requests = 0
        return requests

    def spacebar_status_thread(self):
        if not sys.stdin.isatty():
            self.get_logger().warn("Spacebar status hotkey disabled: stdin is not a TTY.")
            return

        fd = sys.stdin.fileno()
        original_tty = None
        raw_mode_enabled = False
        hotkey_info_printed = False

        try:
            while rclpy.ok():
                hotkey_enabled = self.route_manager.is_moving and (not self.awaiting_user_input)

                if hotkey_enabled and not raw_mode_enabled:
                    original_tty = termios.tcgetattr(fd)
                    tty.setcbreak(fd)
                    raw_mode_enabled = True
                    if not hotkey_info_printed:
                        print("[Hotkey] Press SPACE to print robot status.")
                        hotkey_info_printed = True
                elif (not hotkey_enabled) and raw_mode_enabled:
                    termios.tcsetattr(fd, termios.TCSADRAIN, original_tty)
                    raw_mode_enabled = False

                if not hotkey_enabled:
                    time.sleep(0.05)
                    continue

                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                ch = sys.stdin.read(1)
                if ch == " ":
                    self.queue_status_print()
        except Exception as exc:
            self.get_logger().warn(f"Spacebar status hotkey stopped: {exc}")
        finally:
            if raw_mode_enabled and original_tty is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, original_tty)

    def input_thread(self):
        time.sleep(2)
        print("Available Points:", ", ".join(KEY_POINTS.keys()))
        print("Press SPACE while the robot is moving to print navigation status.")

        print("\n--- Set Initial Robot Pose ---")
        try:
            init_x = float(self.read_user_line("Enter Initial X: "))
            init_y = float(self.read_user_line("Enter Initial Y: "))
            init_yaw_deg = float(self.read_user_line("Enter Initial Yaw (degrees): "))

            yaw_rad = math.radians(init_yaw_deg)
            self.pending_frame_alignment = (init_x, init_y, yaw_rad)

            if not self.try_align_frames():
                self.get_logger().warn(
                    "TF pose unavailable during initialization. Frame alignment will retry automatically."
                )

            init_x_internal, init_y_internal = self.coords.to_internal_xy(init_x, init_y)

            init_msg = PoseWithCovarianceStamped()
            init_msg.header.stamp = self.get_clock().now().to_msg()
            init_msg.header.frame_id = "map"
            init_msg.pose.pose.position.x = init_x_internal
            init_msg.pose.pose.position.y = init_y_internal
            init_msg.pose.pose.orientation.x = 0.0
            init_msg.pose.pose.orientation.y = 0.0
            init_msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
            init_msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
            init_msg.pose.covariance = [
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.06853891945200942,
            ]

            for _ in range(8):
                init_msg.header.stamp = self.get_clock().now().to_msg()
                self.initial_pose_pub.publish(init_msg)
                time.sleep(0.2)

            self.pose_estimator.set_initial_pose_estimate(init_x_internal, init_y_internal, yaw_rad)

            print(
                "Initial pose set (external): ({:.2f}, {:.2f}, {:.1f}deg)".format(
                    init_x, init_y, init_yaw_deg
                )
            )

            wait_start = time.time()
            while (
                rclpy.ok()
                and not self.pose_estimator.odom_received
                and (time.time() - wait_start) < 5.0
            ):
                time.sleep(0.1)

            if self.pose_estimator.set_manual_anchor_from_initial_pose(
                init_x_internal,
                init_y_internal,
                yaw_rad,
            ):
                self.pose_estimator.update_pose_from_manual_anchor()
                ex, ey = self.coords.to_external_xy(
                    self.pose_estimator.current_x,
                    self.pose_estimator.current_y,
                )
                print(
                    "Manual anchor ready. Pose external: ({:.2f}, {:.2f}, {:.1f}deg)".format(
                        ex,
                        ey,
                        math.degrees(self.pose_estimator.current_yaw),
                    )
                )
            else:
                print(
                    "Odometry not received yet. Using TF when available, "
                    "otherwise initial pose temporary."
                )

        except ValueError:
            self.get_logger().error("Invalid initial pose input!")
            return

        while rclpy.ok():
            if self.route_manager.is_moving or self.phase3_pending or self.phase3_active:
                time.sleep(0.1)
                continue

            try:
                user_in = self.read_user_line(
                    "\nEnter Objectives (A;B;DOOR;X,Y): "
                ).strip()
                user_in_upper = user_in.upper()

                if user_in_upper in ["STATUS", "S", "POS", "POSE"]:
                    self.pose_estimator.get_robot_pose()
                    self.telemetry.print_navigation_status(
                        self.pose_estimator,
                        self.local_planner,
                        self.route_manager,
                        "MANUAL STATUS",
                    )
                    continue

                objectives = self.parse_objective_sequence(user_in)
                if not objectives:
                    print("Invalid input.")
                    continue

                if not (self.pose_estimator.get_robot_pose() or self.pose_estimator.initial_pose_received):
                    print("Cannot localize robot in map frame yet.")
                    continue

                current_external_xy = self.coords.to_external_xy(
                    self.pose_estimator.current_x,
                    self.pose_estimator.current_y,
                )
                (
                    mandatory_internal,
                    mandatory_external,
                    door_transition_required,
                    door_internal,
                    phase2_injected,
                ) = self.build_mandatory_route(
                    objectives,
                    current_external_xy,
                )

                if not mandatory_internal:
                    print("Invalid input.")
                    continue

                final_external = objectives[-1]["external"]
                final_pos = self.coords.to_internal_xy(final_external[0], final_external[1])

                now = time.time()
                self.route_manager.set_route(
                    final_pos,
                    mandatory_internal,
                    now,
                    door_transition_required=door_transition_required,
                    door_waypoint=door_internal,
                    speed_mode=("passadis" if phase2_injected else "normal"),
                )
                self.local_planner.reset_for_new_route()

                self._reset_phase3_state()
                if phase2_injected and self.config.phase3_enabled:
                    self.phase2_active = True
                    self.phase2_entered_passadis = False
                    self.phase2_returned_base = False
                    self.phase3_arm_after_phase2 = True
                    self.phase2_start_time = now
                    self.phase2_repeat_count = 0
                else:
                    self.phase2_active = False
                    self.phase2_entered_passadis = False
                    self.phase2_returned_base = False
                    self.phase3_arm_after_phase2 = False
                    self.phase2_repeat_count = 0

                route_steps = " -> ".join(
                    "({:.2f}, {:.2f})".format(p[0], p[1])
                    for p in mandatory_external
                )

                print(
                    "Route set (external): Start -> {}".format(
                        route_steps,
                    )
                )
                if phase2_injected:
                    print("Phase 2 enabled: preset Passadis route injected before final BASE approach.")

                self.telemetry.print_navigation_status(
                    self.pose_estimator,
                    self.local_planner,
                    self.route_manager,
                    "BEFORE FIRST PLAN",
                )

                ok = self.route_manager.start_next_segment(
                    (self.pose_estimator.current_x, self.pose_estimator.current_y),
                    now,
                )
                if not ok:
                    self.telemetry.print_planning_debug(
                        (self.pose_estimator.current_x, self.pose_estimator.current_y),
                        (self.route_manager.target_x, self.route_manager.target_y),
                        self.map_manager,
                        self.global_planner,
                        self.pose_estimator,
                        self.local_planner,
                        self.route_manager,
                        "SEGMENT",
                    )

            except Exception as exc:
                print("Input Error:", exc)
                self.route_manager.is_moving = False

    def control_loop(self):
        if not self.frame_alignment_done and self.pending_frame_alignment is not None:
            self.try_align_frames()

        if not self.pose_estimator.get_robot_pose() and not self.pose_estimator.initial_pose_received:
            return

        now = time.time()
        self.write_runtime_log(now)

        pending_status_prints = self.consume_status_print_requests()
        for _ in range(pending_status_prints):
            self.telemetry.print_navigation_status(
                self.pose_estimator,
                self.local_planner,
                self.route_manager,
                "MANUAL STATUS",
            )

        if now - self.last_marker_publish_time >= self.config.marker_publish_period:
            self.telemetry.publish_debug_markers(
                self.debug_marker_pub,
                self.pose_estimator,
                self.route_manager,
            )
            self.telemetry.publish_planned_path(
                self.path_pub,
                self.pose_estimator,
                self.route_manager,
            )
            self.last_marker_publish_time = now

        self._maybe_start_phase3(now)

        move = self.local_planner.step(now, self.pose_estimator, self.route_manager)
        move.header.stamp = self.get_clock().now().to_msg()
        move.header.frame_id = "base_link"
        self.publisher.publish(move)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        node.publisher.publish(stop)
        node.close_runtime_log_file()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
