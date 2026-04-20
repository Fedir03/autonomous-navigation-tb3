import math

from geometry_msgs.msg import TwistStamped

from .config import normalize_angle


class LocalPlanner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # LiDAR sectors
        self.min_front_dist = float("inf")
        self.min_left_dist = float("inf")
        self.min_right_dist = float("inf")

        # FSM state
        self.nav_state = "FOLLOW_GOAL"  # FOLLOW_GOAL | AVOID_OBSTACLE | WALL_FOLLOW | REJOIN_PATH | BACKUP | DOOR_ADVANCE_Y | DOOR_TURN_LEFT | DOOR_CROSS_X
        self.wall_follow_side = "right"
        self.wall_follow_hit_distance = float("inf")
        self.bug2_line_start = None
        self.bug2_line_goal = None
        self.preferred_escape_heading = None
        self.wall_follow_enter_time = 0.0

        self.follow_obstacle_hit_count = 0
        self.follow_obstacle_hit_last_time = 0.0

        self.door_ref_yaw = 0.0
        self.door_turn_target_yaw = 0.0
        self.door_segment_start_time = 0.0
        self.door_segment_start_xy = None

    def reset_for_new_route(self):
        self.nav_state = "FOLLOW_GOAL"
        self.wall_follow_side = "right"
        self.wall_follow_hit_distance = float("inf")
        self.bug2_line_start = None
        self.bug2_line_goal = None
        self.preferred_escape_heading = None
        self.wall_follow_enter_time = 0.0
        self.follow_obstacle_hit_count = 0
        self.follow_obstacle_hit_last_time = 0.0

        self.door_ref_yaw = 0.0
        self.door_turn_target_yaw = 0.0
        self.door_segment_start_time = 0.0
        self.door_segment_start_xy = None

    def _start_door_transition(self, current_x, current_y, current_yaw, now):
        self.nav_state = "DOOR_ADVANCE_Y"
        self.door_ref_yaw = current_yaw
        self.door_turn_target_yaw = normalize_angle(current_yaw + (math.pi / 2.0))
        self.door_segment_start_time = now
        self.door_segment_start_xy = (current_x, current_y)

    def _step_door_transition(self, move, now, current_x, current_y, current_yaw, route_manager):
        if self.nav_state == "DOOR_ADVANCE_Y":
            yaw_err = normalize_angle(self.door_ref_yaw - current_yaw)
            move.twist.linear.x = min(self.config.door_follow_speed, self.config.max_speed * 0.8)
            move.twist.angular.z = max(min(yaw_err * self.config.door_heading_kp, 0.8), -0.8)

            enough_forward_time = (now - self.door_segment_start_time) >= self.config.door_min_forward_time_before_left_check
            if enough_forward_time and self.min_left_dist > self.config.door_left_opening_distance:
                self.nav_state = "DOOR_TURN_LEFT"
            return move

        if self.nav_state == "DOOR_TURN_LEFT":
            yaw_err = normalize_angle(self.door_turn_target_yaw - current_yaw)
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(min(yaw_err * self.config.kp_angular, self.config.door_turn_speed), -self.config.door_turn_speed)

            if abs(yaw_err) < self.config.door_turn_tolerance:
                self.nav_state = "DOOR_CROSS_X"
                self.door_segment_start_xy = (current_x, current_y)
            return move

        # DOOR_CROSS_X
        yaw_err = normalize_angle(self.door_turn_target_yaw - current_yaw)
        move.twist.linear.x = min(self.config.door_follow_speed, self.config.max_speed * 0.8)
        move.twist.angular.z = max(min(yaw_err * self.config.door_heading_kp, 0.8), -0.8)

        if self.door_segment_start_xy is not None:
            dx = current_x - self.door_segment_start_xy[0]
            dy = current_y - self.door_segment_start_xy[1]
            if math.hypot(dx, dy) >= self.config.door_cross_x_distance:
                self.nav_state = "FOLLOW_GOAL"
                route_manager.complete_door_transition((current_x, current_y), now)
        return move

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

    def get_secondary_axis_heading(self, current_x, current_y, target_x, target_y):
        dx = target_x - current_x
        dy = target_y - current_y

        if abs(dx) >= abs(dy):
            if abs(dy) < 1e-3:
                return None
            return math.pi / 2.0 if dy >= 0.0 else -math.pi / 2.0

        if abs(dx) < 1e-3:
            return None
        return 0.0 if dx >= 0.0 else math.pi

    def choose_wall_follow_side(self, current_x, current_y, current_yaw, target_x, target_y):
        desired_heading = self.get_secondary_axis_heading(current_x, current_y, target_x, target_y)
        self.preferred_escape_heading = desired_heading

        if desired_heading is None:
            self.wall_follow_side = "right" if self.min_right_dist <= self.min_left_dist else "left"
            return

        yaw_err = normalize_angle(desired_heading - current_yaw)
        side_from_heading = "left" if yaw_err >= 0.0 else "right"

        if side_from_heading == "left":
            if self.min_left_dist + 0.08 >= self.min_right_dist:
                self.wall_follow_side = "left"
            else:
                self.wall_follow_side = "right"
        else:
            if self.min_right_dist + 0.08 >= self.min_left_dist:
                self.wall_follow_side = "right"
            else:
                self.wall_follow_side = "left"

    def enter_wall_follow(self, current_x, current_y, current_yaw, target_x, target_y, now):
        self.wall_follow_hit_distance = math.hypot(target_x - current_x, target_y - current_y)
        self.bug2_line_start = (current_x, current_y)
        self.bug2_line_goal = (target_x, target_y)
        self.wall_follow_enter_time = now
        self.choose_wall_follow_side(current_x, current_y, current_yaw, target_x, target_y)

    def point_to_line_distance(self, p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        den = math.hypot(bx - ax, by - ay)
        if den < 1e-6:
            return math.hypot(px - ax, py - ay)
        num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        return num / den

    def line_of_sight_clear(self, threshold=0.8):
        return self.min_front_dist > max(self.config.safe_stop_distance, threshold)

    def _handle_avoid_obstacle(self, move: TwistStamped):
        move.twist.linear.x = 0.0
        turn_sign = -1.0 if self.wall_follow_side == "right" else 1.0
        move.twist.angular.z = turn_sign * self.config.avoid_turn_speed

    def _distance_to(self, x0, y0, x1, y1):
        return math.hypot(x1 - x0, y1 - y0)

    def step(self, now, pose_estimator, route_manager):
        move = TwistStamped()

        current_x = pose_estimator.current_x
        current_y = pose_estimator.current_y
        current_yaw = pose_estimator.current_yaw

        target = route_manager.target

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

        if self.nav_state in ["DOOR_ADVANCE_Y", "DOOR_TURN_LEFT", "DOOR_CROSS_X"]:
            # Keep hard front safety even during door transition maneuver.
            if self.min_front_dist < self.config.safe_stop_distance and self.nav_state != "DOOR_TURN_LEFT":
                move.twist.linear.x = 0.0
                move.twist.angular.z = 0.0
                return move
            return self._step_door_transition(move, now, current_x, current_y, current_yaw, route_manager)

        # If there is no active path, retry pending segments and keep local escape active if needed.
        if (not route_manager.is_moving) or (not route_manager.path):
            if route_manager.pending_segment_target is not None and now >= route_manager.next_segment_retry_time:
                ok = route_manager.start_next_segment((current_x, current_y), now)
                if (not ok) and route_manager.replan_fail_streak >= 3 and route_manager.target is not None:
                    self.nav_state = "WALL_FOLLOW"
                    self.enter_wall_follow(
                        current_x,
                        current_y,
                        current_yaw,
                        route_manager.target_x,
                        route_manager.target_y,
                        now,
                    )

            if self.nav_state in ["WALL_FOLLOW", "AVOID_OBSTACLE", "REJOIN_PATH", "BACKUP", "DOOR_ADVANCE_Y", "DOOR_TURN_LEFT", "DOOR_CROSS_X"] and (route_manager.target is not None or route_manager.door_transition_active):
                route_manager.is_moving = True
            else:
                return move

        target = route_manager.target
        if target is None:
            return move

        # Hard safety while following path
        if self.nav_state == "FOLLOW_GOAL" and self.min_front_dist < self.config.safe_stop_distance:
            self.nav_state = "AVOID_OBSTACLE"

        if self.min_front_dist > self.config.follow_block_trigger_distance:
            self.follow_obstacle_hit_count = 0

        # Progress watchdog while in FOLLOW_GOAL
        if self.nav_state == "FOLLOW_GOAL":
            dist_goal = route_manager.distance_to_active_goal((current_x, current_y))
            if dist_goal is not None:
                if dist_goal < route_manager.best_distance_on_segment - self.config.progress_epsilon:
                    route_manager.best_distance_on_segment = dist_goal
                    route_manager.last_progress_time = now
                elif now - route_manager.last_progress_time > self.config.stuck_timeout:
                    self.logger.warn("Stuck detected. Replan then Bug2 fallback.")
                    if not route_manager.try_replan_current_segment((current_x, current_y), now):
                        self.nav_state = "WALL_FOLLOW"
                        self.enter_wall_follow(current_x, current_y, current_yaw, target[0], target[1], now)
                    route_manager.last_progress_time = now

        if self.nav_state == "AVOID_OBSTACLE":
            self.choose_wall_follow_side(current_x, current_y, current_yaw, target[0], target[1])
            self._handle_avoid_obstacle(move)
            if self.min_front_dist > self.config.caution_distance:
                self.nav_state = "WALL_FOLLOW"
                self.enter_wall_follow(current_x, current_y, current_yaw, target[0], target[1], now)
            return move

        if self.nav_state == "WALL_FOLLOW":
            if self.min_front_dist < self.config.safe_stop_distance:
                self._handle_avoid_obstacle(move)
            else:
                side_dist = self.min_right_dist if self.wall_follow_side == "right" else self.min_left_dist
                pref_err = None
                if self.preferred_escape_heading is not None:
                    pref_err = normalize_angle(self.preferred_escape_heading - current_yaw)

                if not math.isfinite(side_dist):
                    if pref_err is None:
                        turn_sign = -1.0 if self.wall_follow_side == "right" else 1.0
                        move.twist.linear.x = min(self.config.wall_follow_speed * 0.5, self.config.max_speed * 0.4)
                        move.twist.angular.z = turn_sign * 0.4
                    else:
                        move.twist.linear.x = min(self.config.wall_follow_speed * 0.8, self.config.max_speed * 0.45)
                        move.twist.angular.z = max(min(pref_err * 1.2, 0.9), -0.9)
                else:
                    err = self.config.wall_follow_distance - side_dist
                    sign = -1.0 if self.wall_follow_side == "right" else 1.0
                    ang_cmd = sign * self.config.wall_follow_kp * err

                    if pref_err is not None:
                        ang_cmd += max(min(pref_err * 0.5, 0.35), -0.35)

                    if pref_err is not None and abs(pref_err) > 0.9:
                        move.twist.linear.x = 0.0
                    else:
                        move.twist.linear.x = min(self.config.wall_follow_speed, self.config.max_speed * 0.5)
                    move.twist.angular.z = max(min(ang_cmd, 0.9), -0.9)

            # Opportunistic exit: if there is frontal clearance, try to recover A* path.
            if self.line_of_sight_clear(0.8) and route_manager.try_replan_current_segment((current_x, current_y), now):
                self.nav_state = "FOLLOW_GOAL"
                return move

            if self.bug2_line_start is not None and self.bug2_line_goal is not None:
                d_line = self.point_to_line_distance(
                    (current_x, current_y), self.bug2_line_start, self.bug2_line_goal
                )
                d_goal = math.hypot(target[0] - current_x, target[1] - current_y)
                enough_time = (now - self.wall_follow_enter_time) > self.config.wall_follow_min_time
                if (
                    enough_time
                    and d_line < 0.20
                    and d_goal < (self.wall_follow_hit_distance - 0.20)
                    and self.line_of_sight_clear(0.8)
                ):
                    self.nav_state = "REJOIN_PATH"
            return move

        if self.nav_state == "REJOIN_PATH":
            if route_manager.try_replan_current_segment((current_x, current_y), now):
                self.nav_state = "FOLLOW_GOAL"
            elif now - route_manager.last_replan_attempt_time >= self.config.replan_cooldown:
                self.nav_state = "WALL_FOLLOW"
            return move

        # FOLLOW_GOAL + A*
        if self.min_front_dist < self.config.follow_block_trigger_distance:
            if now - self.follow_obstacle_hit_last_time > self.config.follow_obstacle_hit_reset_s:
                self.follow_obstacle_hit_count = 0
            self.follow_obstacle_hit_count += 1
            self.follow_obstacle_hit_last_time = now

            self.logger.warn("Obstacle ahead during FOLLOW_GOAL. Replan/fallback.")
            if self.follow_obstacle_hit_count >= self.config.follow_obstacle_hit_threshold:
                self.nav_state = "WALL_FOLLOW"
                self.enter_wall_follow(current_x, current_y, current_yaw, target[0], target[1], now)
                self.logger.warn("Persistent frontal block. Switching to WALL_FOLLOW escape.")
            elif not route_manager.try_replan_current_segment((current_x, current_y), now):
                self.nav_state = "AVOID_OBSTACLE"
            return move

        if route_manager.current_wp_index >= len(route_manager.path):
            self.logger.info("Segment reached.")
            if route_manager.begin_door_transition_if_needed():
                self._start_door_transition(current_x, current_y, current_yaw, now)
                return move
            route_manager.start_next_segment((current_x, current_y), now)
            return move

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
            return move

        if abs(yaw_err) > self.config.yaw_stop_threshold:
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(min(yaw_err * self.config.kp_angular, 1.0), -1.0)
        else:
            max_allowed = self.config.max_speed if self.min_front_dist > self.config.caution_distance else self.config.wall_follow_speed
            ang_ratio = min(abs(yaw_err) / max(self.config.yaw_stop_threshold, 1e-6), 1.0)
            speed_scale = 1.0 - (0.75 * ang_ratio)
            lin_cmd = min(dist * self.config.kp_linear, max_allowed) * speed_scale

            if lin_cmd > 0.0:
                lin_cmd = max(lin_cmd, min(self.config.min_motion_linear_speed, max_allowed))

            move.twist.linear.x = lin_cmd
            move.twist.angular.z = max(min(yaw_err * self.config.kp_angular, 1.0), -1.0)

        return move
