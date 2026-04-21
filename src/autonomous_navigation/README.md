# Autonomous Navigation (ROS 2 Jazzy)

## 1) Project Methodology

The project follows a safety-first iterative process:

1. Define a baseline global-local navigation loop.
2. Add obstacle handling with clear priority rules (safety override before tracking).
3. Validate in runtime and simplify logic when behavior becomes unstable.
4. Keep changes traceable through small, focused modules.

The architecture is modular:

- `main_node.py`: orchestration, ROS I/O, control loop.
- `global_planner.py`: A* planning over occupancy maps.
- `local_planner.py`: path tracking + safety overrides + door transition state machine.
- `route_manager.py`: segment/waypoint progression and replanning lifecycle.
- `map_manager.py`: map/base_map selection and occupancy access.
- `pose_estimator.py`: TF/odometry pose source handling.
- `telemetry.py`: status output and RViz visualization.
- `config.py`: centralized tuning and thresholds.

## 2) Engineering Standards

### Code quality

- Code is organized by responsibility and avoids monolithic files.
- Formatting pass applied to improve readability and long-line control.
- Runtime syntax checks pass with:

```bash
python3 -m py_compile autonomous_navigation/*.py autonomous_navigation/launch/*.py
```

### Technical documentation

- This file documents architecture, methodology, safety behavior, and limits.
- Runtime logs are generated automatically (`autonomous_navigation/logs/`) for traceability.

### Maintainability conventions

- Parameters are centralized in `NavigationConfig`.
- Safety constraints are explicit and prioritized over path tracking.
- Planner map handling uses resolution-independent distances in meters.

## 3) Navigation Quality

### Path safety

- Global A* uses obstacle inflation in meters (`inflation_radius_m`) to preserve safety margins across map resolutions.
- Start/goal are snapped to nearest valid free cells within a bounded search radius in meters.
- If base-map planning fails, fallback to SLAM map planning is available.

### Distance/time efficiency

- Path is sampled with configurable waypoint spacing.
- Local tracking uses lookahead pursuit and angular-aware speed scaling.
- Replanning is rate-limited with cooldown to avoid control oscillations.

## 4) Perception and Avoidance (LiDAR)

LiDAR sectors are reduced to `front/left/right` minimum distances.

Safety priority order:

1. Emergency reverse if frontal distance is below backup threshold.
2. Hard stop clamps if frontal distance is below collision threshold.
3. If path is blocked, trigger replanning from current pose.
4. If replanning fails, execute turn-away avoidance toward safer side.
5. Return to path following only when clearance and replanning conditions are met.

This design ensures unknown obstacles can override predefined paths in real time.

## 5) Localization Accuracy Status

Current implementation uses TF (`map -> base_link`) with odometry-anchor fallback.

- Final-pose precision and charging-station docking accuracy are not yet fully implemented as a dedicated subsystem.
- This is an identified next milestone.

## 6) Zero-Collision Design Notes

Absolute collision guarantees are not possible in all real-world conditions, but the system is designed to minimize risk with:

- Multi-threshold distance gating.
- Motion command safety clamps.
- Dynamic replanning when obstacles appear.
- Conservative speed limits near obstacles.

For final evaluation, keep conservative tuning and validate in the target environment before timed runs.
