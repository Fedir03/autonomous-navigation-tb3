import math
import threading
import time

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

        self.map_manager = MapManager()
        self.pose_estimator = PoseEstimator(self)
        self.global_planner = GlobalPlanner(self.map_manager, self.config.inflation_radius)
        self.route_manager = RouteManager(self.global_planner, self.config, self.get_logger())
        self.local_planner = LocalPlanner(self.config, self.get_logger())
        self.telemetry = Telemetry(self, self.coords, KEY_POINTS)

        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, "/nav_debug_markers", 10)
        self.path_pub = self.create_publisher(Path, "/nav_planned_path", 10)

        self.create_subscription(Odometry, "/odom", self.pose_estimator.odom_callback, qos_best_effort)
        self.create_subscription(LaserScan, "/scan", self.local_planner.scan_callback, qos_best_effort)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_map)
        self.create_subscription(OccupancyGrid, "/base_map", self.base_map_callback, qos_map)

        self.last_status_print_time = 0.0
        self.last_marker_publish_time = 0.0

        self.timer = self.create_timer(0.1, self.control_loop)
        threading.Thread(target=self.input_thread, daemon=True).start()

    def map_callback(self, msg: OccupancyGrid):
        self.map_manager.map_callback(msg, self.get_logger())

    def base_map_callback(self, msg: OccupancyGrid):
        self.map_manager.base_map_callback(msg, self.get_logger())

    def input_thread(self):
        time.sleep(2)
        print("Available Points:", ", ".join(KEY_POINTS.keys()))

        print("\n--- Set Initial Robot Pose ---")
        try:
            init_x = float(input("Enter Initial X: "))
            init_y = float(input("Enter Initial Y: "))
            init_yaw_deg = float(input("Enter Initial Yaw (degrees): "))

            yaw_rad = math.radians(init_yaw_deg)

            # Align user/teacher map coordinates to the live SLAM map frame.
            if self.pose_estimator.get_robot_pose():
                self.coords.set_frame_alignment(
                    init_x,
                    init_y,
                    yaw_rad,
                    self.pose_estimator.current_x,
                    self.pose_estimator.current_y,
                    self.pose_estimator.current_yaw,
                )
                self.get_logger().info(
                    "Coordinate frames aligned: external initial pose mapped to current SLAM pose."
                )
            else:
                self.get_logger().warn(
                    "TF pose unavailable during initialization. Using unaligned external coordinates as fallback."
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
            while rclpy.ok() and not self.pose_estimator.odom_received and (time.time() - wait_start) < 5.0:
                time.sleep(0.1)

            if self.pose_estimator.set_manual_anchor_from_initial_pose(init_x_internal, init_y_internal, yaw_rad):
                self.pose_estimator.update_pose_from_manual_anchor()
                ex, ey = self.coords.to_external_xy(self.pose_estimator.current_x, self.pose_estimator.current_y)
                print(
                    "Manual anchor ready. Pose external: ({:.2f}, {:.2f}, {:.1f}deg)".format(
                        ex,
                        ey,
                        math.degrees(self.pose_estimator.current_yaw),
                    )
                )
            else:
                print("Odometry not received yet. Using TF when available, otherwise initial pose temporary.")

        except ValueError:
            self.get_logger().error("Invalid initial pose input!")
            return

        while rclpy.ok():
            if not (self.map_manager.map_received or self.map_manager.base_map_received):
                print("Waiting for /map or /base_map...")
                time.sleep(1)
                continue

            try:
                user_in = input("\nEnter Final Target (Name or X,Y): ").strip().upper()
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
                if user_in in KEY_POINTS:
                    final_pos = self.coords.to_internal_xy(KEY_POINTS[user_in][0], KEY_POINTS[user_in][1])
                else:
                    parts = user_in.split(",")
                    if len(parts) == 2:
                        final_pos = self.coords.to_internal_xy(float(parts[0]), float(parts[1]))

                if not final_pos:
                    print("Invalid input.")
                    continue

                if not (self.pose_estimator.get_robot_pose() or self.pose_estimator.initial_pose_received):
                    print("Cannot localize robot in map frame yet.")
                    continue

                b_wp = self.coords.to_internal_xy(KEY_POINTS["B"][0], KEY_POINTS["B"][1])
                o_wp = self.coords.to_internal_xy(KEY_POINTS["O"][0], KEY_POINTS["O"][1])

                now = time.time()
                self.route_manager.set_route(final_pos, [b_wp, o_wp, final_pos], now)
                self.local_planner.reset_for_new_route()

                print(
                    "Route set (external): Start -> B {} -> O {} -> Target {}".format(
                        KEY_POINTS["B"],
                        KEY_POINTS["O"],
                        self.coords.format_external_xy(final_pos),
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
        if not self.pose_estimator.get_robot_pose() and not self.pose_estimator.initial_pose_received:
            return

        now = time.time()
        if now - self.last_status_print_time >= self.config.status_print_period:
            self.telemetry.print_navigation_status(
                self.pose_estimator,
                self.local_planner,
                self.route_manager,
                "RUN",
            )
            self.last_status_print_time = now

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
