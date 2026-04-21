import heapq
import math


class GlobalPlanner:
    def __init__(self, map_manager, inflation_radius=4, waypoint_spacing=0.30):
        self.map_manager = map_manager
        self.inflation_radius = inflation_radius
        self.waypoint_spacing = waypoint_spacing
        self.last_error = ""

    def _bresenham_cells(self, start_cell, end_cell):
        x0, y0 = start_cell
        x1, y1 = end_cell

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            yield (x, y)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _segment_is_free(self, a_cell, b_cell):
        for gx, gy in self._bresenham_cells(a_cell, b_cell):
            if not self.is_cell_free(gx, gy):
                return False
        return True

    def _simplify_grid_path(self, path_grid):
        if len(path_grid) <= 2:
            return list(path_grid)

        simplified = [path_grid[0]]
        anchor_idx = 0

        while anchor_idx < len(path_grid) - 1:
            furthest_idx = anchor_idx + 1
            probe_idx = furthest_idx
            while probe_idx < len(path_grid):
                if self._segment_is_free(path_grid[anchor_idx], path_grid[probe_idx]):
                    furthest_idx = probe_idx
                    probe_idx += 1
                else:
                    break

            simplified.append(path_grid[furthest_idx])
            anchor_idx = furthest_idx

        return simplified

    def is_cell_free(self, gx, gy):
        if not self.map_manager.in_bounds(gx, gy):
            return False

        occ = self.map_manager.get_cell_occupancy(gx, gy)
        if occ > 50:
            return False

        r = self.inflation_radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if self.map_manager.in_bounds(nx, ny):
                    if self.map_manager.get_cell_occupancy(nx, ny) > 50:
                        return False
        return True

    def find_nearest_free_cell(self, gx, gy, max_radius=25):
        width, height, _, _ = self.map_manager.get_active_map_info()
        if width <= 0 or height <= 0:
            return None

        gx, gy = self.map_manager.clamp_to_map(gx, gy)
        if self.is_cell_free(gx, gy):
            return (gx, gy)

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if self.map_manager.in_bounds(nx, ny) and self.is_cell_free(nx, ny):
                        return (nx, ny)
        return None

    def _try_slam_fallback(self, start_xy, end_xy, reason):
        if self.map_manager.active_map_source() != "base" or (not self.map_manager.map_received):
            self.last_error = reason
            return []

        prev_preference = self.map_manager.prefer_base_map_for_planning
        self.map_manager.prefer_base_map_for_planning = False
        try:
            fallback_path = self.calculate_path(start_xy, end_xy)
        finally:
            self.map_manager.prefer_base_map_for_planning = prev_preference

        if fallback_path:
            self.last_error = ""
            return fallback_path

        self.last_error = reason + " (SLAM fallback also failed)"
        return []

    def calculate_path(self, start_xy, end_xy):
        width, height, resolution, origin = self.map_manager.get_active_map_info()
        if width <= 0 or height <= 0:
            self.last_error = "No map/base_map received yet."
            return []

        start_grid = self.map_manager.world_to_grid(*start_xy)
        end_grid = self.map_manager.world_to_grid(*end_xy)
        if start_grid is None or end_grid is None:
            return self._try_slam_fallback(start_xy, end_xy, "Invalid map or coordinates.")

        if (not self.map_manager.in_bounds(start_grid[0], start_grid[1])) or (
            not self.map_manager.in_bounds(end_grid[0], end_grid[1])
        ):
            min_x = origin[0]
            min_y = origin[1]
            max_x = origin[0] + width * resolution
            max_y = origin[1] + height * resolution
            reason = (
                f"Requested start/goal outside map bounds. "
                f"Map x:[{min_x:.2f},{max_x:.2f}] y:[{min_y:.2f},{max_y:.2f}]"
            )
            return self._try_slam_fallback(start_xy, end_xy, reason)

        start_grid = self.find_nearest_free_cell(*start_grid)
        end_grid = self.find_nearest_free_cell(*end_grid)
        if start_grid is None or end_grid is None:
            return self._try_slam_fallback(start_xy, end_xy, "Could not find nearby free start/end cell.")

        frontier = []
        heapq.heappush(frontier, (0.0, start_grid))
        came_from = {start_grid: None}
        cost_so_far = {start_grid: 0.0}
        found = False

        while frontier:
            current = heapq.heappop(frontier)[1]
            if current == end_grid:
                found = True
                break

            for dx, dy in [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.map_manager.in_bounds(neighbor[0], neighbor[1]) and self.is_cell_free(*neighbor):
                    step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    new_cost = cost_so_far[current] + step_cost
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + math.hypot(end_grid[0] - neighbor[0], end_grid[1] - neighbor[1])
                        heapq.heappush(frontier, (priority, neighbor))
                        came_from[neighbor] = current

        if not found:
            return self._try_slam_fallback(start_xy, end_xy, "A* failed to find a path with current map(s).")

        path_grid = []
        curr = end_grid
        while curr != start_grid:
            path_grid.append(curr)
            curr = came_from[curr]
        path_grid.reverse()

        simplified_grid = path_grid

        world_path = []
        for cell in simplified_grid:
            wp = self.map_manager.grid_to_world(*cell)
            if not world_path:
                world_path.append(wp)
                continue

            if math.hypot(wp[0] - world_path[-1][0], wp[1] - world_path[-1][1]) >= self.waypoint_spacing:
                world_path.append(wp)

        final_wp = self.map_manager.grid_to_world(*end_grid)
        sampled = list(world_path)
        if not sampled:
            sampled = [final_wp]
        elif sampled[-1] != final_wp:
            sampled.append(final_wp)

        self.last_error = ""
        return sampled
