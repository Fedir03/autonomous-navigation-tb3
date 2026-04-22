import math

from geometry_msgs.msg import TwistStamped

from .config import normalize_angle


class LocalPlanner:
    def __init__(self, config, logger, coord_adapter):
        self.config = config
        self.logger = logger
        self.coords = coord_adapter

        # LiDAR sectors
        self.min_front_dist = float("inf")
        self.min_left_dist = float("inf")
        self.min_right_dist = float("inf")

        # FSM state
        # FOLLOW_GOAL | AVOID_OBSTACLE (Bug2 wall-follow) | BACKUP |
        # DOOR_ALIGN_ZERO | DOOR_SEARCH_LEFT | DOOR_ALIGN_NINETY | DOOR_CROSS
        self.nav_state = "FOLLOW_GOAL"
        self.turn_away_sign = 1.0

        # Bug2 context (obstacle contour follow + leave-to-goal-line condition)
        self.bug2_line_start_xy = None
        self.bug2_goal_xy = None
        self.bug2_hit_distance_to_goal = float("inf")
        self.bug2_follow_left_wall = True

        self.door_target_yaw_zero = 0.0
        self.door_target_yaw_ninety = 0.0
        self.door_search_start_time = 0.0
        self.door_cross_start_xy = None
        self.door_left_opening_seen = False

    def reset_for_new_route(self):
        self.nav_state = "FOLLOW_GOAL"
        self.turn_away_sign = 1.0
        self.bug2_line_start_xy = None
        self.bug2_goal_xy = None
        self.bug2_hit_distance_to_goal = float("inf")
        self.bug2_follow_left_wall = True
        self.door_target_yaw_zero = 0.0
        self.door_target_yaw_ninety = 0.0
        self.door_search_start_time = 0.0
        self.door_cross_start_xy = None
        self.door_left_opening_seen = False

    def scan_callback(self, msg):
        try:
            n = len(msg.ranges)
            if n == 0:
                return

            front = msg.ranges[0:15] + msg.ranges[-15:]
            left = msg.ranges[60:100] if n >= 100 else msg.ranges[n // 4 : n // 3]
            right = msg.ranges[260:300] if n >= 300 else msg.ranges[2 * n // 3 : 3 * n // 4]

            valid_front = [r for r in front if msg.range_min < r < msg.range_max]
            valid_left = [r for r in left if msg.range_min < r < msg.range_max]
            valid_right = [r for r in right if msg.range_min < r < msg.range_max]

            self.min_front_dist = min(valid_front) if valid_front else float("inf")
            self.min_left_dist = min(valid_left) if valid_left else float("inf")
            self.min_right_dist = min(valid_right) if valid_right else float("inf")
        except Exception:
            pass

    def _handle_avoid_obstacle(self, move: TwistStamped):
        move.twist.linear.x = 0.0
        move.twist.angular.z = self.turn_away_sign * self.config.avoid_turn_speed

    def _distance_point_to_line(self, point, line_start, line_end):
        if line_start is None or line_end is None:
            return float("inf")

        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        denom = math.hypot(x2 - x1, y2 - y1)
        if denom < 1e-6:
            return math.hypot(x0 - x1, y0 - y1)

        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 * y1) - (y2 * x1))
        return num / denom

    def _enter_bug2_follow(self, current_x, current_y, target_xy):
        self.nav_state = "AVOID_OBSTACLE"
        self.bug2_line_start_xy = (current_x, current_y)
        self.bug2_goal_xy = target_xy
        self.bug2_hit_distance_to_goal = math.hypot(
            target_xy[0] - current_x,
            target_xy[1] - current_y,
        )

        # Keep the obstacle on the tighter side and move through the freer side.
        self.bug2_follow_left_wall = self.min_left_dist < self.min_right_dist
        self.turn_away_sign = 1.0 if self.min_left_dist > self.min_right_dist else -1.0

    def _should_leave_bug2(self, current_x, current_y):
        if self.bug2_goal_xy is None or self.bug2_line_start_xy is None:
            return False

        line_dist = self._distance_point_to_line(
            (current_x, current_y),
            self.bug2_line_start_xy,
            self.bug2_goal_xy,
        )
        goal_dist = math.hypot(
            self.bug2_goal_xy[0] - current_x,
            self.bug2_goal_xy[1] - current_y,
        )

        on_m_line = line_dist <= self.config.bug2_mline_tolerance
        made_progress = goal_dist <= (
            self.bug2_hit_distance_to_goal - self.config.bug2_leave_progress
        )
        enough_front_clearance = self.min_front_dist > self.config.caution_distance
        return on_m_line and made_progress and enough_front_clearance

    def _step_bug2_follow(self, move, current_x, current_y, current_yaw):
        # If obstacle is still directly ahead, pivot to slide around it.
        if self.min_front_dist < self.config.follow_block_trigger_distance:
            move.twist.linear.x = 0.0
            move.twist.angular.z = (
                -self.config.avoid_turn_speed
                if self.bug2_follow_left_wall
                else self.config.avoid_turn_speed
            )
            return move

        desired = self.config.wall_follow_distance
        side_dist = self.min_left_dist if self.bug2_follow_left_wall else self.min_right_dist
        if not math.isfinite(side_dist):
            side_dist = desired * 1.8

        side_error = side_dist - desired
        side_correction = side_error * self.config.wall_follow_kp
        if not self.bug2_follow_left_wall:
            side_correction = -side_correction

        # Keep bias toward goal direction so wall-follow is not aimless.
        if self.bug2_goal_xy is not None:
            goal_heading = math.atan2(self.bug2_goal_xy[1] - current_y, self.bug2_goal_xy[0] - current_x)
            goal_heading_err = normalize_angle(goal_heading - current_yaw)
            goal_correction = max(min(goal_heading_err * 0.35, 0.3), -0.3)
        else:
            goal_correction = 0.0

        move.twist.linear.x = self.config.wall_follow_speed
        move.twist.angular.z = max(
            min(side_correction + goal_correction, self.config.avoid_turn_speed),
            -self.config.avoid_turn_speed,
        )
        return move

    def _apply_safety_clamps(self, move: TwistStamped):
        # Conservative stop when frontal clearance is already in warning range.
        if move.twist.linear.x > 0.0 and self.min_front_dist < self.config.safe_stop_distance:
            move.twist.linear.x = 0.0

        # Never command forward motion when frontal clearance is critically low.
        if move.twist.linear.x > 0.0 and self.min_front_dist < self.config.collision_stop_distance:
            move.twist.linear.x = 0.0

        # Avoid sweeping into nearby side obstacles while rotating.
        if (
            abs(move.twist.angular.z) > 0.2
            and min(self.min_left_dist, self.min_right_dist) < self.config.turn_side_clearance
        ):
            move.twist.linear.x = 0.0
            if self.min_left_dist < self.min_right_dist:
                move.twist.angular.z = min(move.twist.angular.z, 0.0)
            else:
                move.twist.angular.z = max(move.twist.angular.z, 0.0)

        return move

    def _start_door_transition(self, now, current_x, current_y):
        self.nav_state = "DOOR_ALIGN_ZERO"
        self.door_target_yaw_zero = self.coords.external_yaw_to_internal(0.0)
        self.door_target_yaw_ninety = self.coords.external_yaw_to_internal(
            math.pi / 2.0
        )
        self.door_search_start_time = now
        self.door_cross_start_xy = (current_x, current_y)
        self.door_left_opening_seen = False

    def _step_door_transition(
        self,
        move,
        now,
        current_x,
        current_y,
        current_yaw,
        route_manager,
    ):
        if self.nav_state == "DOOR_ALIGN_ZERO":
            yaw_err = normalize_angle(self.door_target_yaw_zero - current_yaw)
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(min(yaw_err * self.config.kp_angular, 0.9), -0.9)
            if abs(yaw_err) <= self.config.door_align_tolerance:
                self.nav_state = "DOOR_SEARCH_LEFT"
                self.door_search_start_time = now
                self.door_left_opening_seen = False
            return move

        if self.nav_state == "DOOR_SEARCH_LEFT":
            yaw_err = normalize_angle(self.door_target_yaw_zero - current_yaw)
            move.twist.linear.x = self.config.door_forward_speed
            move.twist.angular.z = max(
                min(yaw_err * self.config.door_heading_kp, 0.6),
                -0.6,
            )

            if (now - self.door_search_start_time) >= self.config.door_search_min_time:
                if self.min_left_dist > self.config.door_left_opening_distance:
                    self.door_left_opening_seen = True

            ext_x, _ = self.coords.to_external_xy(current_x, current_y)
            center_reached = (
                abs(ext_x - self.config.door_left_center_x)
                <= self.config.door_left_center_x_tolerance
            )
            center_passed = ext_x > (
                self.config.door_left_center_x + self.config.door_left_center_x_tolerance
            )
            if self.door_left_opening_seen and (center_reached or center_passed):
                self.nav_state = "DOOR_ALIGN_NINETY"
            return move

        if self.nav_state == "DOOR_ALIGN_NINETY":
            yaw_err = normalize_angle(self.door_target_yaw_ninety - current_yaw)
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(
                min(yaw_err * self.config.kp_angular, 0.9),
                -0.9,
            )
            if abs(yaw_err) <= self.config.door_align_tolerance:
                self.nav_state = "DOOR_CROSS"
                self.door_cross_start_xy = (current_x, current_y)
            return move

        # DOOR_CROSS
        yaw_err = normalize_angle(self.door_target_yaw_ninety - current_yaw)
        move.twist.linear.x = self.config.door_forward_speed
        move.twist.angular.z = max(min(yaw_err * self.config.door_heading_kp, 0.6), -0.6)

        if self.door_cross_start_xy is not None:
            d = math.hypot(current_x - self.door_cross_start_xy[0], current_y - self.door_cross_start_xy[1])
            if d >= self.config.door_second_cross_distance:
                self.nav_state = "FOLLOW_GOAL"
                route_manager.complete_door_transition((current_x, current_y), now)
        return move

    def _distance_to(self, x0, y0, x1, y1):
        return math.hypot(x1 - x0, y1 - y0)

    def step(self, now, pose_estimator, route_manager):
        move = TwistStamped()

        current_x = pose_estimator.current_x
        current_y = pose_estimator.current_y
        current_yaw = pose_estimator.current_yaw

        # Hard-priority emergency reverse when something is too close in front.
        if self.nav_state != "BACKUP" and self.min_front_dist < self.config.backup_min_front_distance:
            self.nav_state = "BACKUP"

        if self.nav_state == "BACKUP":
            if self.min_front_dist < self.config.backup_min_front_distance:
                move.twist.linear.x = -abs(self.config.backup_speed)
                move.twist.angular.z = 0.0
                return move

            # Recovered clearance: continue with normal obstacle handling.
            self.nav_state = "AVOID_OBSTACLE"

        if self.nav_state in ["DOOR_ALIGN_ZERO", "DOOR_SEARCH_LEFT", "DOOR_ALIGN_NINETY", "DOOR_CROSS"]:
            if (
                self.min_front_dist < self.config.safe_stop_distance
                and self.nav_state != "DOOR_ALIGN_NINETY"
            ):
                move.twist.linear.x = 0.0
                move.twist.angular.z = 0.0
                return self._apply_safety_clamps(move)
            return self._apply_safety_clamps(
                self._step_door_transition(move, now, current_x, current_y, current_yaw, route_manager)
            )

        # If there is no active path, retry pending segments.
        if (not route_manager.is_moving) or (not route_manager.path):
            if (
                route_manager.pending_segment_target is not None
                and now >= route_manager.next_segment_retry_time
            ):
                route_manager.start_next_segment((current_x, current_y), now)

            if not route_manager.path:
                return move

        target = route_manager.target
        if target is None:
            return move

        if self.nav_state == "AVOID_OBSTACLE":
            self._step_bug2_follow(move, current_x, current_y, current_yaw)
            replan_ok = route_manager.try_replan_current_segment((current_x, current_y), now)
            if self._should_leave_bug2(current_x, current_y) and (replan_ok or self.min_front_dist > self.config.caution_distance):
                self.nav_state = "FOLLOW_GOAL"
                self.bug2_line_start_xy = None
                self.bug2_goal_xy = None
                self.bug2_hit_distance_to_goal = float("inf")
                self.logger.info("Bug2 leave condition met. Returning to FOLLOW_GOAL.")
            return self._apply_safety_clamps(move)

        # FOLLOW_GOAL + simple obstacle handling
        if self.min_front_dist < self.config.follow_block_trigger_distance:
            if not route_manager.try_replan_current_segment((current_x, current_y), now):
                self._enter_bug2_follow(current_x, current_y, target)
            return self._apply_safety_clamps(move)

        if route_manager.current_wp_index >= len(route_manager.path):
            self.logger.info("Segment reached.")
            if route_manager.begin_door_transition_if_needed():
                self._start_door_transition(now, current_x, current_y)
                return self._apply_safety_clamps(move)
            route_manager.start_next_segment((current_x, current_y), now)
            return self._apply_safety_clamps(move)

        while route_manager.current_wp_index < (len(route_manager.path) - 1):
            cwp_x, cwp_y = route_manager.path[route_manager.current_wp_index]
            if self._distance_to(current_x, current_y, cwp_x, cwp_y) < self.config.xy_tolerance:
                route_manager.current_wp_index += 1
            else:
                break

        if route_manager.current_wp_index >= len(route_manager.path):
            return move

        pursuit_index = route_manager.current_wp_index
        while pursuit_index < (len(route_manager.path) - 1):
            p_x, p_y = route_manager.path[pursuit_index]
            if self._distance_to(current_x, current_y, p_x, p_y) < self.config.follow_lookahead_distance:
                pursuit_index += 1
            else:
                break

        wp_x, wp_y = route_manager.path[pursuit_index]
        dx = wp_x - current_x
        dy = wp_y - current_y
        dist = math.hypot(dx, dy)
        yaw_err = normalize_angle(math.atan2(dy, dx) - current_yaw)

        if pursuit_index == (len(route_manager.path) - 1) and dist < self.config.xy_tolerance:
            route_manager.current_wp_index += 1
            return self._apply_safety_clamps(move)

        if abs(yaw_err) > self.config.yaw_stop_threshold:
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(min(yaw_err * self.config.kp_angular, 1.0), -1.0)
        else:
            max_allowed = (
                self.config.max_speed
                if self.min_front_dist > self.config.caution_distance
                else self.config.wall_follow_speed
            )
            ang_ratio = min(abs(yaw_err) / max(self.config.yaw_stop_threshold, 1e-6), 1.0)
            speed_scale = 1.0 - (0.75 * ang_ratio)
            lin_cmd = min(dist * self.config.kp_linear, max_allowed) * speed_scale

            if lin_cmd > 0.0:
                lin_cmd = max(lin_cmd, min(self.config.min_motion_linear_speed, max_allowed))

            move.twist.linear.x = lin_cmd
            move.twist.angular.z = max(min(yaw_err * self.config.kp_angular, 1.0), -1.0)

        return self._apply_safety_clamps(move)
