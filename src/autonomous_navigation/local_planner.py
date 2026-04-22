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
        # FOLLOW_GOAL | AVOID_OBSTACLE (pivot bypass) | BACKUP |
        # DOOR_ALIGN_ZERO | DOOR_SEARCH_LEFT | DOOR_ALIGN_NINETY | DOOR_CROSS
        self.nav_state = "FOLLOW_GOAL"
        self.turn_away_sign = 1.0

        # Simple obstacle bypass context.
        self.avoid_stage = None
        self.avoid_side_sign = 1.0
        self.avoid_pivot_xy = None
        self.avoid_search_start_time = 0.0
        self.avoid_search_cycles = 0
        self.avoid_retry_not_before = 0.0

        self.door_target_yaw_zero = 0.0
        self.door_target_yaw_ninety = 0.0
        self.door_search_start_time = 0.0
        self.door_cross_start_xy = None
        self.door_left_opening_seen = False

    def reset_for_new_route(self):
        self.nav_state = "FOLLOW_GOAL"
        self.turn_away_sign = 1.0
        self.avoid_stage = None
        self.avoid_side_sign = 1.0
        self.avoid_pivot_xy = None
        self.avoid_search_start_time = 0.0
        self.avoid_search_cycles = 0
        self.avoid_retry_not_before = 0.0
        self.door_target_yaw_zero = 0.0
        self.door_target_yaw_ninety = 0.0
        self.door_search_start_time = 0.0
        self.door_cross_start_xy = None
        self.door_left_opening_seen = False

    def _min_distance_in_sector(self, msg, center_deg, width_deg):
        if not msg.ranges:
            return float("inf")
        if abs(msg.angle_increment) < 1e-9:
            return float("inf")

        center = math.radians(center_deg)
        half = math.radians(width_deg * 0.5)
        best = float("inf")

        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            ang = msg.angle_min + (i * msg.angle_increment)
            if abs(normalize_angle(ang - center)) <= half:
                if r < best:
                    best = r

        return best

    def scan_callback(self, msg):
        try:
            if len(msg.ranges) == 0:
                return

            # Front cone widened to 100 degrees as requested.
            self.min_front_dist = self._min_distance_in_sector(
                msg,
                0.0,
                self.config.lidar_front_cone_deg,
            )
            self.min_left_dist = self._min_distance_in_sector(
                msg,
                90.0,
                self.config.lidar_side_cone_deg,
            )
            self.min_right_dist = self._min_distance_in_sector(
                msg,
                -90.0,
                self.config.lidar_side_cone_deg,
            )
        except Exception:
            pass

    def _clear_avoid_state(self):
        self.avoid_stage = None
        self.avoid_pivot_xy = None
        self.avoid_search_start_time = 0.0
        self.avoid_search_cycles = 0

    def _is_direct_clear(self, route_manager, start_xy, end_xy, unknown_as_blocked=False):
        planner = route_manager.planner
        if hasattr(planner, "is_direct_segment_clear"):
            return planner.is_direct_segment_clear(
                start_xy,
                end_xy,
                unknown_as_blocked=unknown_as_blocked,
            )
        return False

    def _find_pivot_candidate(self, current_x, current_y, current_yaw, target_xy, route_manager, preferred_side):
        side_candidates = [preferred_side, -preferred_side]
        current_goal_dist = math.hypot(target_xy[0] - current_x, target_xy[1] - current_y)

        for side in side_candidates:
            side_clearance = self.min_left_dist if side > 0 else self.min_right_dist
            if side_clearance < self.config.turn_side_clearance:
                continue

            for angle_deg in self.config.avoid_pivot_angle_candidates_deg:
                heading = current_yaw + (side * math.radians(angle_deg))
                for d in self.config.avoid_pivot_distance_candidates:
                    px = current_x + d * math.cos(heading)
                    py = current_y + d * math.sin(heading)
                    pivot = (px, py)

                    if not self._is_direct_clear(
                        route_manager,
                        (current_x, current_y),
                        pivot,
                        unknown_as_blocked=False,
                    ):
                        continue

                    pivot_goal_dist = math.hypot(target_xy[0] - px, target_xy[1] - py)
                    if pivot_goal_dist > (current_goal_dist + self.config.avoid_goal_distance_slack):
                        continue

                    # Prefer direct-clear pivots, but allow riskier pivots that improve
                    # progress when direct line to goal is still partially obstructed.
                    if self._is_direct_clear(
                        route_manager,
                        pivot,
                        target_xy,
                        unknown_as_blocked=False,
                    ):
                        return pivot, side

                    # Accept near-goal-progress pivots even when pivot->goal line
                    # is not fully clear yet; local bypass + forced replan handles
                    # the continuation.
                    return pivot, side

        return None, preferred_side

    def _start_pivot_avoid(self, now, current_x, current_y, current_yaw, target_xy, route_manager):
        preferred_side = 1.0 if self.min_left_dist > self.min_right_dist else -1.0
        pivot, side = self._find_pivot_candidate(
            current_x,
            current_y,
            current_yaw,
            target_xy,
            route_manager,
            preferred_side,
        )

        self.nav_state = "AVOID_OBSTACLE"
        self.avoid_side_sign = side
        self.avoid_search_start_time = now
        self.avoid_search_cycles = 0

        if pivot is not None:
            self.avoid_pivot_xy = pivot
            self.avoid_stage = "TURN_TO_PIVOT"
            self.logger.info(
                "Obstacle bypass: pivot found at ({:.2f}, {:.2f}).".format(
                    pivot[0],
                    pivot[1],
                )
            )
        else:
            self.avoid_pivot_xy = None
            self.avoid_stage = "SEARCH_PIVOT"
            self.logger.info("Obstacle bypass: no pivot yet, searching.")

    def _step_pivot_avoid(self, move, now, current_x, current_y, current_yaw, target_xy, route_manager):
        if target_xy is None:
            return move

        if (
            self.min_front_dist > self.config.follow_block_trigger_distance
            and self._is_direct_clear(
                route_manager,
                (current_x, current_y),
                target_xy,
                unknown_as_blocked=False,
            )
        ):
            route_manager.try_replan_current_segment((current_x, current_y), now, force=True)
            self.nav_state = "FOLLOW_GOAL"
            self._clear_avoid_state()
            self.logger.info("Obstacle bypass complete: direct objective line is clear.")
            return move

        if self.avoid_stage == "SEARCH_PIVOT":
            pivot, side = self._find_pivot_candidate(
                current_x,
                current_y,
                current_yaw,
                target_xy,
                route_manager,
                self.avoid_side_sign,
            )
            if pivot is not None:
                self.avoid_pivot_xy = pivot
                self.avoid_side_sign = side
                self.avoid_stage = "TURN_TO_PIVOT"
                return move

            move.twist.linear.x = 0.0
            move.twist.angular.z = self.avoid_side_sign * (self.config.avoid_turn_speed * 0.8)

            if (now - self.avoid_search_start_time) > self.config.avoid_search_timeout:
                forced_replan_ok = route_manager.try_replan_current_segment(
                    (current_x, current_y),
                    now,
                    force=True,
                )
                if forced_replan_ok and self.min_front_dist > self.config.follow_block_trigger_distance:
                    self.nav_state = "FOLLOW_GOAL"
                    self._clear_avoid_state()
                    self.logger.info("Fallback replan succeeded after pivot search.")
                    return move

                self.avoid_search_cycles += 1
                self.avoid_side_sign *= -1.0
                self.avoid_search_start_time = now
                if self.avoid_search_cycles >= self.config.avoid_max_search_cycles:
                    self.logger.warn("Pivot search exhausted. Triggering BACKUP.")
                    self.nav_state = "BACKUP"
                    self._clear_avoid_state()
            return move

        if self.avoid_stage == "TURN_TO_PIVOT":
            if self.avoid_pivot_xy is None:
                self.avoid_stage = "SEARCH_PIVOT"
                self.avoid_search_start_time = now
                return move

            yaw_target = math.atan2(
                self.avoid_pivot_xy[1] - current_y,
                self.avoid_pivot_xy[0] - current_x,
            )
            yaw_err = normalize_angle(yaw_target - current_yaw)
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(
                min(yaw_err * self.config.kp_angular, self.config.avoid_turn_speed),
                -self.config.avoid_turn_speed,
            )
            if abs(yaw_err) <= self.config.avoid_turn_tolerance:
                self.avoid_stage = "MOVE_TO_PIVOT"
            return move

        if self.avoid_stage == "MOVE_TO_PIVOT":
            if self.avoid_pivot_xy is None:
                self.avoid_stage = "SEARCH_PIVOT"
                self.avoid_search_start_time = now
                return move

            if self.min_front_dist < self.config.safe_stop_distance:
                self.avoid_stage = "SEARCH_PIVOT"
                self.avoid_search_start_time = now
                return move

            dx = self.avoid_pivot_xy[0] - current_x
            dy = self.avoid_pivot_xy[1] - current_y
            dist = math.hypot(dx, dy)
            if dist <= self.config.avoid_pivot_reach_tolerance:
                self.avoid_stage = "TURN_TO_GOAL"
                return move

            yaw_err = normalize_angle(math.atan2(dy, dx) - current_yaw)
            if abs(yaw_err) > self.config.yaw_stop_threshold:
                move.twist.linear.x = 0.0
            else:
                lin = min(self.config.avoid_forward_speed, dist * self.config.kp_linear)
                if lin > 0.0:
                    lin = max(lin, min(self.config.min_motion_linear_speed, self.config.avoid_forward_speed))
                move.twist.linear.x = lin

            move.twist.angular.z = max(
                min(yaw_err * self.config.kp_angular, self.config.avoid_turn_speed),
                -self.config.avoid_turn_speed,
            )
            return move

        if self.avoid_stage == "TURN_TO_GOAL":
            yaw_goal = math.atan2(target_xy[1] - current_y, target_xy[0] - current_x)
            yaw_err = normalize_angle(yaw_goal - current_yaw)
            move.twist.linear.x = 0.0
            move.twist.angular.z = max(
                min(yaw_err * self.config.kp_angular, self.config.avoid_turn_speed),
                -self.config.avoid_turn_speed,
            )
            if abs(yaw_err) <= self.config.avoid_turn_tolerance:
                route_manager.try_replan_current_segment((current_x, current_y), now, force=True)
                self.nav_state = "FOLLOW_GOAL"
                self._clear_avoid_state()
                self.logger.info("Obstacle bypass complete: returning to FOLLOW_GOAL.")
            return move

        self.avoid_stage = "SEARCH_PIVOT"
        self.avoid_search_start_time = now
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
            if self.min_front_dist < self.config.backup_recover_distance:
                move.twist.linear.x = -abs(self.config.backup_speed)
                move.twist.angular.z = 0.0
                return move

            # Recovered clearance: continue with normal obstacle handling.
            self.nav_state = "FOLLOW_GOAL"
            self._clear_avoid_state()
            self.avoid_retry_not_before = now + self.config.avoid_retry_cooldown

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
            self._step_pivot_avoid(
                move,
                now,
                current_x,
                current_y,
                current_yaw,
                target,
                route_manager,
            )
            return self._apply_safety_clamps(move)

        # FOLLOW_GOAL + simple obstacle handling
        if self.min_front_dist < self.config.follow_block_trigger_distance:
            if now < self.avoid_retry_not_before:
                move.twist.linear.x = 0.0
                move.twist.angular.z = (
                    self.config.avoid_turn_speed * 0.5
                    if self.min_left_dist > self.min_right_dist
                    else -self.config.avoid_turn_speed * 0.5
                )
                return self._apply_safety_clamps(move)

            replan_ok = route_manager.try_replan_current_segment((current_x, current_y), now)
            if not replan_ok:
                self._start_pivot_avoid(
                    now,
                    current_x,
                    current_y,
                    current_yaw,
                    target,
                    route_manager,
                )
                self._step_pivot_avoid(
                    move,
                    now,
                    current_x,
                    current_y,
                    current_yaw,
                    target,
                    route_manager,
                )
                return self._apply_safety_clamps(move)
            # Replan succeeded: continue below to follow the new path immediately
            # (typically starts with an in-place rotation), instead of returning
            # early with zero commands.

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
