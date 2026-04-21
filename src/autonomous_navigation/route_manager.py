import math


class RouteManager:
    def __init__(self, planner, config, logger):
        self.planner = planner
        self.config = config
        self.logger = logger

        self.target_x = None
        self.target_y = None
        self.final_target = None
        self.path = []
        self.current_wp_index = 0
        self.is_moving = False
        self.global_waypoints = []
        self.pending_segment_target = None

        self.next_segment_retry_time = 0.0
        self.replan_count = 0
        self.replan_fail_streak = 0
        self.last_replan_attempt_time = 0.0

        self.best_distance_on_segment = float("inf")
        self.last_progress_time = 0.0

        self.door_transition_required = False
        self.door_transition_done = False
        self.door_transition_active = False
        self.door_waypoint = None

    @property
    def target(self):
        if self.target_x is None or self.target_y is None:
            return None
        return (self.target_x, self.target_y)

    def _nearest_waypoint_index(self, current_xy):
        if not self.path:
            return 0

        cx, cy = current_xy
        best_idx = 0
        best_dist = float("inf")
        for idx, wp in enumerate(self.path):
            d = math.hypot(wp[0] - cx, wp[1] - cy)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def set_route(
        self,
        final_target,
        mandatory_waypoints,
        now,
        door_transition_required=False,
        door_waypoint=None,
    ):
        self.final_target = final_target
        self.global_waypoints = list(mandatory_waypoints)
        self.pending_segment_target = None
        self.next_segment_retry_time = 0.0

        self.target_x = None
        self.target_y = None
        self.path = []
        self.current_wp_index = 0
        self.is_moving = False

        self.replan_fail_streak = 0
        self.best_distance_on_segment = float("inf")
        self.last_progress_time = now

        self.door_transition_required = door_transition_required
        self.door_transition_done = False
        self.door_transition_active = False
        self.door_waypoint = door_waypoint

    def clear_route(self):
        self.target_x = None
        self.target_y = None
        self.final_target = None
        self.path = []
        self.current_wp_index = 0
        self.is_moving = False
        self.global_waypoints = []
        self.pending_segment_target = None
        self.next_segment_retry_time = 0.0

        self.door_transition_required = False
        self.door_transition_done = False
        self.door_transition_active = False
        self.door_waypoint = None

    def begin_door_transition_if_needed(self):
        if (
            (not self.door_transition_required)
            or self.door_transition_done
            or self.door_transition_active
        ):
            return False

        if self.target is None or self.door_waypoint is None:
            return False

        at_door_segment = (
            math.hypot(
                self.target_x - self.door_waypoint[0],
                self.target_y - self.door_waypoint[1],
            )
            < 0.30
        )
        if not at_door_segment:
            return False

        # Only perform transition when there are pending waypoints beyond DOOR.
        if len(self.global_waypoints) == 0:
            return False

        self.door_transition_active = True
        self.path = []
        self.current_wp_index = 0
        self.is_moving = True
        self.logger.info("Starting DOOR transition maneuver.")
        return True

    def complete_door_transition(self, current_xy, now):
        if not self.door_transition_active:
            return False

        self.door_transition_active = False
        self.door_transition_done = True
        self.target_x = None
        self.target_y = None
        self.path = []
        self.current_wp_index = 0
        self.logger.info("DOOR transition completed. Continuing route.")
        return self.start_next_segment(current_xy, now)

    def start_next_segment(self, current_xy, now):
        if self.pending_segment_target is None and not self.global_waypoints:
            self.clear_route()
            print("All segments completed!")
            return True

        if self.pending_segment_target is None:
            self.pending_segment_target = self.global_waypoints[0]

        target = self.pending_segment_target
        self.target_x, self.target_y = target

        print("Planning segment to (internal): ({:.2f}, {:.2f})".format(self.target_x, self.target_y))
        self.path = self.planner.calculate_path(current_xy, target)

        if self.path:
            if self.global_waypoints and self.global_waypoints[0] == self.pending_segment_target:
                self.global_waypoints.pop(0)
            self.pending_segment_target = None
            self.next_segment_retry_time = 0.0
            self.current_wp_index = self._nearest_waypoint_index(current_xy)
            self.is_moving = True
            self.replan_fail_streak = 0
            self.best_distance_on_segment = math.hypot(
                self.target_x - current_xy[0],
                self.target_y - current_xy[1],
            )
            self.last_progress_time = now
            print("Path plotted with {} waypoints.".format(len(self.path)))
            return True

        reason = self.planner.last_error if self.planner.last_error else "Unknown planning failure"
        print("Failed to plan path to segment target. Retrying in 2s... Reason: {}".format(reason))
        self.is_moving = False
        self.replan_fail_streak += 1
        self.next_segment_retry_time = now + 2.0
        return False

    def try_replan_current_segment(self, current_xy, now):
        if self.target is None:
            return False

        if now - self.last_replan_attempt_time < self.config.replan_cooldown:
            return False
        self.last_replan_attempt_time = now

        self.replan_count += 1
        new_path = self.planner.calculate_path(current_xy, self.target)
        if new_path:
            self.path = new_path
            self.current_wp_index = self._nearest_waypoint_index(current_xy)
            self.is_moving = True
            self.replan_fail_streak = 0
            self.best_distance_on_segment = math.hypot(
                self.target_x - current_xy[0],
                self.target_y - current_xy[1],
            )
            self.last_progress_time = now
            self.logger.info("Replan success. Returning to FOLLOW_GOAL.")
            return True

        self.replan_fail_streak += 1
        return False

    def distance_to_active_goal(self, current_xy):
        if self.target is None:
            return None
        return math.hypot(self.target_x - current_xy[0], self.target_y - current_xy[1])

    def distance_to_final_goal(self, current_xy):
        if self.final_target is None:
            return None
        return math.hypot(self.final_target[0] - current_xy[0], self.final_target[1] - current_xy[1])
