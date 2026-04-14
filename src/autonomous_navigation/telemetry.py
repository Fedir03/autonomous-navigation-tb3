import math

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

from .config import quaternion_from_yaw


class Telemetry:
    def __init__(self, node, coord_adapter, key_points):
        self.node = node
        self.coord_adapter = coord_adapter
        self.key_points = key_points

    def print_planning_debug(
        self,
        start_xy,
        end_xy,
        map_manager,
        planner,
        pose_estimator,
        local_planner,
        route_manager,
        context_label="PLAN",
    ):
        print("\n--- Planning Debug [{}] ---".format(context_label))
        width, height, resolution, origin = map_manager.get_active_map_info()
        if width <= 0 or height <= 0:
            print("Map: not received")
            return

        min_x = origin[0]
        min_y = origin[1]
        max_x = origin[0] + width * resolution
        max_y = origin[1] + height * resolution

        start_grid = map_manager.world_to_grid(*start_xy)
        end_grid = map_manager.world_to_grid(*end_xy)

        print(
            "Map bounds: x:[{:.2f}, {:.2f}] y:[{:.2f}, {:.2f}] res:{:.3f} size:{}x{}".format(
                min_x, max_x, min_y, max_y, resolution, width, height
            )
        )
        print(
            "Robot pose internal: ({:.2f}, {:.2f}, {:.1f}deg)".format(
                pose_estimator.current_x,
                pose_estimator.current_y,
                math.degrees(pose_estimator.current_yaw),
            )
        )
        print(
            "Requested (external): start={} goal={}".format(
                self.coord_adapter.format_external_xy(start_xy),
                self.coord_adapter.format_external_xy(end_xy),
            )
        )
        print("Requested (internal): start={} goal={}".format(start_xy, end_xy))
        print("Grid: start={} goal={}".format(start_grid, end_grid))

        if start_grid is not None:
            s_in = map_manager.in_bounds(start_grid[0], start_grid[1])
            s_free = planner.is_cell_free(*start_grid) if s_in else False
            print("Start in_bounds={} free={}".format(s_in, s_free))
        if end_grid is not None:
            g_in = map_manager.in_bounds(end_grid[0], end_grid[1])
            g_free = planner.is_cell_free(*end_grid) if g_in else False
            print("Goal in_bounds={} free={}".format(g_in, g_free))

        print(
            "LiDAR front/left/right = {:.2f}/{:.2f}/{:.2f} m".format(
                local_planner.min_front_dist,
                local_planner.min_left_dist,
                local_planner.min_right_dist,
            )
        )
        print(
            "State={} replans={} fail_streak={}".format(
                local_planner.nav_state,
                route_manager.replan_count,
                route_manager.replan_fail_streak,
            )
        )
        print("Last planner error: {}".format(planner.last_error if planner.last_error else "(none)"))

    def print_navigation_status(self, pose_estimator, local_planner, route_manager, label="STATUS"):
        active_goal = route_manager.target
        final_goal = route_manager.final_target if route_manager.final_target is not None else active_goal

        dist_active = None
        dist_final = None
        if active_goal is not None:
            dist_active = math.hypot(active_goal[0] - pose_estimator.current_x, active_goal[1] - pose_estimator.current_y)
        if final_goal is not None:
            dist_final = math.hypot(final_goal[0] - pose_estimator.current_x, final_goal[1] - pose_estimator.current_y)

        ex, ey = self.coord_adapter.to_external_xy(pose_estimator.current_x, pose_estimator.current_y)

        print("\n--- {} ---".format(label))
        print("Pose estimada (externa): x={:.2f}, y={:.2f}, yaw={:.1f}deg".format(ex, ey, math.degrees(pose_estimator.current_yaw)))
        print("Pose interna planner: x={:.2f}, y={:.2f}".format(pose_estimator.current_x, pose_estimator.current_y))
        print("Fuente de pose: {}".format(pose_estimator.pose_source))
        print("Objetivo activo (externo): {}".format(self.coord_adapter.format_external_xy(active_goal)))
        print("Objetivo final (externo): {}".format(self.coord_adapter.format_external_xy(final_goal)))
        if dist_active is not None:
            print("Distancia al objetivo activo: {:.2f} m".format(dist_active))
        if dist_final is not None:
            print("Distancia al objetivo final: {:.2f} m".format(dist_final))
        print("Waypoint actual: {}/{}".format(route_manager.current_wp_index, len(route_manager.path)))
        print(
            "LiDAR front/left/right: {:.2f}/{:.2f}/{:.2f} m".format(
                local_planner.min_front_dist,
                local_planner.min_left_dist,
                local_planner.min_right_dist,
            )
        )
        print(
            "Estado nav: {} | Replans: {} | Fail streak: {}".format(
                local_planner.nav_state,
                route_manager.replan_count,
                route_manager.replan_fail_streak,
            )
        )

    def publish_debug_markers(self, marker_pub, pose_estimator, route_manager):
        markers = MarkerArray()
        now = self.node.get_clock().now().to_msg()

        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = now
        clear.ns = "nav_debug"
        clear.id = 0
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        marker_id = 1

        for name, xy in self.key_points.items():
            kx, ky = self.coord_adapter.to_internal_xy(xy[0], xy[1])

            m_point = Marker()
            m_point.header.frame_id = "map"
            m_point.header.stamp = now
            m_point.ns = "key_points"
            m_point.id = marker_id
            marker_id += 1
            m_point.type = Marker.SPHERE
            m_point.action = Marker.ADD
            m_point.pose.position.x = float(kx)
            m_point.pose.position.y = float(ky)
            m_point.pose.position.z = 0.05
            m_point.pose.orientation.w = 1.0
            m_point.scale.x = 0.15
            m_point.scale.y = 0.15
            m_point.scale.z = 0.15
            m_point.color.r = 0.95
            m_point.color.g = 0.2
            m_point.color.b = 0.2
            m_point.color.a = 0.9
            markers.markers.append(m_point)

            m_text = Marker()
            m_text.header.frame_id = "map"
            m_text.header.stamp = now
            m_text.ns = "key_points_text"
            m_text.id = marker_id
            marker_id += 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(kx)
            m_text.pose.position.y = float(ky)
            m_text.pose.position.z = 0.35
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.20
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 0.1
            m_text.color.a = 0.95
            m_text.text = name
            markers.markers.append(m_text)

        m_robot = Marker()
        m_robot.header.frame_id = "map"
        m_robot.header.stamp = now
        m_robot.ns = "robot"
        m_robot.id = marker_id
        marker_id += 1
        m_robot.type = Marker.ARROW
        m_robot.action = Marker.ADD
        m_robot.pose.position.x = float(pose_estimator.current_x)
        m_robot.pose.position.y = float(pose_estimator.current_y)
        m_robot.pose.position.z = 0.06
        qx, qy, qz, qw = quaternion_from_yaw(pose_estimator.current_yaw)
        m_robot.pose.orientation.x = qx
        m_robot.pose.orientation.y = qy
        m_robot.pose.orientation.z = qz
        m_robot.pose.orientation.w = qw
        m_robot.scale.x = 0.45
        m_robot.scale.y = 0.12
        m_robot.scale.z = 0.12
        m_robot.color.r = 0.1
        m_robot.color.g = 0.9
        m_robot.color.b = 0.1
        m_robot.color.a = 0.95
        markers.markers.append(m_robot)

        if route_manager.target is not None:
            m_active = Marker()
            m_active.header.frame_id = "map"
            m_active.header.stamp = now
            m_active.ns = "goal_active"
            m_active.id = marker_id
            marker_id += 1
            m_active.type = Marker.SPHERE
            m_active.action = Marker.ADD
            m_active.pose.position.x = float(route_manager.target_x)
            m_active.pose.position.y = float(route_manager.target_y)
            m_active.pose.position.z = 0.07
            m_active.pose.orientation.w = 1.0
            m_active.scale.x = 0.22
            m_active.scale.y = 0.22
            m_active.scale.z = 0.22
            m_active.color.r = 0.1
            m_active.color.g = 0.5
            m_active.color.b = 1.0
            m_active.color.a = 0.95
            markers.markers.append(m_active)

        if route_manager.final_target is not None:
            m_final = Marker()
            m_final.header.frame_id = "map"
            m_final.header.stamp = now
            m_final.ns = "goal_final"
            m_final.id = marker_id
            marker_id += 1
            m_final.type = Marker.CYLINDER
            m_final.action = Marker.ADD
            m_final.pose.position.x = float(route_manager.final_target[0])
            m_final.pose.position.y = float(route_manager.final_target[1])
            m_final.pose.position.z = 0.05
            m_final.pose.orientation.w = 1.0
            m_final.scale.x = 0.28
            m_final.scale.y = 0.28
            m_final.scale.z = 0.06
            m_final.color.r = 1.0
            m_final.color.g = 0.85
            m_final.color.b = 0.1
            m_final.color.a = 0.95
            markers.markers.append(m_final)

        if route_manager.path:
            m_path = Marker()
            m_path.header.frame_id = "map"
            m_path.header.stamp = now
            m_path.ns = "planned_path"
            m_path.id = marker_id
            m_path.type = Marker.LINE_STRIP
            m_path.action = Marker.ADD
            m_path.pose.orientation.w = 1.0
            m_path.scale.x = 0.05
            m_path.color.r = 0.1
            m_path.color.g = 1.0
            m_path.color.b = 1.0
            m_path.color.a = 0.9

            p0 = Point()
            p0.x = float(pose_estimator.current_x)
            p0.y = float(pose_estimator.current_y)
            p0.z = 0.04
            m_path.points.append(p0)

            for wp in route_manager.path:
                p = Point()
                p.x = float(wp[0])
                p.y = float(wp[1])
                p.z = 0.04
                m_path.points.append(p)

            markers.markers.append(m_path)

        marker_pub.publish(markers)

    def publish_planned_path(self, path_pub, pose_estimator, route_manager):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()

        start = PoseStamped()
        start.header = msg.header
        start.pose.position.x = float(pose_estimator.current_x)
        start.pose.position.y = float(pose_estimator.current_y)
        start.pose.orientation.w = 1.0
        msg.poses.append(start)

        for wp in route_manager.path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        path_pub.publish(msg)
