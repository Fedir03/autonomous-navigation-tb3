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
from .telemetry import Telemetry


class PointAToBNode(Node):
    def __init__(self):
        super().__init__("point_a_to_b_node")

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
            self.config.path_min_waypoint_spacing,
        )
        self.route_manager = RouteManager(self.global_planner, self.config, self.get_logger())
        self.local_planner = LocalPlanner(self.config, self.get_logger(), self.coords)
        self.telemetry = Telemetry(self, self.coords, KEY_POINTS)

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
        self.create_subscription(LaserScan, "/scan", self.local_planner.scan_callback, qos_best_effort)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_map)
        self.create_subscription(OccupancyGrid, "/base_map", self.base_map_callback, qos_map)
        # Fallback QoS to receive map topics when publisher QoS differs.
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

        self.timer = self.create_timer(0.1, self.control_loop)
        threading.Thread(target=self.input_thread, daemon=True).start()
        threading.Thread(target=self.spacebar_status_thread, daemon=True).start()

    def _open_runtime_log_file(self):
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.runtime_log_path = os.path.join(logs_dir, f"navigation_runtime_{stamp}.csv")
        self.runtime_log_file = open(self.runtime_log_path, "w", encoding="utf-8", buffering=1)
        self.runtime_log_file.write("timestamp,current_phase,robot_x,robot_y,robot_yaw\n")
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

    def write_runtime_log(self, now):
        if self.runtime_log_file is None:
            return
        if (now - self.last_runtime_log_time) < 1.0:
            return

        ex, ey = self.coords.to_external_xy(self.pose_estimator.current_x, self.pose_estimator.current_y)
        eyaw = self.coords.internal_yaw_to_external(self.pose_estimator.current_yaw)
        self._update_phase_from_external_y(ey)

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.runtime_log_file.write(
            f"{ts},{self.current_phase},{ex:.3f},{ey:.3f},{eyaw:.4f}\n"
        )
        self.last_runtime_log_time = now

    def map_callback(self, msg: OccupancyGrid):
        self.map_manager.map_callback(msg, self.get_logger())

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
        # Convert the calibrated external anchor of base-map origin to current map frame.
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

    def build_mandatory_route(self, target_external, target_name=None):
        tx, ty = target_external
        route_external = []

        forced_by_name = (
            target_name in self.config.door_forced_targets
            if target_name is not None
            else False
        )
        needs_door = forced_by_name or (ty > self.config.door_required_y_threshold)
        door_external = KEY_POINTS["DOOR"]

        if needs_door:
            route_external.append(door_external)

        route_external.append((tx, ty))

        route_internal = []
        compact_external = []
        for ext_wp in route_external:
            int_wp = self.coords.to_internal_xy(ext_wp[0], ext_wp[1])
            if route_internal:
                if math.hypot(int_wp[0] - route_internal[-1][0], int_wp[1] - route_internal[-1][1]) < 0.05:
                    continue
            route_internal.append(int_wp)
            compact_external.append(ext_wp)

        is_door_target = (target_name == "DOOR") or (
            math.hypot(tx - door_external[0], ty - door_external[1]) < 0.30
        )
        is_original_room_target = (
            target_name in ("A", "B", "C", "D", "E")
            if target_name is not None
            else False
        )
        door_transition_required = (
            needs_door and (not is_door_target) and (not is_original_room_target)
        )

        door_internal = (
            self.coords.to_internal_xy(door_external[0], door_external[1])
            if needs_door
            else None
        )
        return route_internal, compact_external, door_transition_required, door_internal

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

            # Try alignment now; if TF is not available yet, retry automatically in control loop.
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
            if self.route_manager.is_moving:
                time.sleep(0.1)
                continue

            try:
                user_in = self.read_user_line("\nEnter Final Target (Name or X,Y): ").strip().upper()
                if user_in in ["STATUS", "S", "POS", "POSE"]:
                    self.pose_estimator.get_robot_pose()
                    self.telemetry.print_navigation_status(
                        self.pose_estimator,
                        self.local_planner,
                        self.route_manager,
                        "MANUAL STATUS",
                    )
                    continue

                final_pos = None
                final_external = None
                target_name = None
                if user_in in KEY_POINTS:
                    final_external = KEY_POINTS[user_in]
                    target_name = user_in
                else:
                    parts = user_in.split(",")
                    if len(parts) == 2:
                        final_external = (float(parts[0]), float(parts[1]))

                if final_external is not None:
                    final_pos = self.coords.to_internal_xy(final_external[0], final_external[1])

                if not final_pos:
                    print("Invalid input.")
                    continue

                if not (self.pose_estimator.get_robot_pose() or self.pose_estimator.initial_pose_received):
                    print("Cannot localize robot in map frame yet.")
                    continue

                (
                    mandatory_internal,
                    mandatory_external,
                    door_transition_required,
                    door_internal,
                ) = self.build_mandatory_route(
                    final_external,
                    target_name,
                )

                now = time.time()
                self.route_manager.set_route(
                    final_pos,
                    mandatory_internal,
                    now,
                    door_transition_required=door_transition_required,
                    door_waypoint=door_internal,
                )
                self.local_planner.reset_for_new_route()

                route_steps = " -> ".join(
                    "({:.2f}, {:.2f})".format(p[0], p[1])
                    for p in mandatory_external
                )

                print(
                    "Route set (external): Start -> {}".format(
                        route_steps,
                    )
                )

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
        # Keep trying frame alignment until TF map pose becomes available.
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

        move = self.local_planner.step(now, self.pose_estimator, self.route_manager)
        move.header.stamp = self.get_clock().now().to_msg()
        move.header.frame_id = "base_link"
        self.publisher.publish(move)


def main(args=None):
    print("Initializing A* + Bug2 Navigation Node...")
    rclpy.init(args=args)
    node = PointAToBNode()
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
