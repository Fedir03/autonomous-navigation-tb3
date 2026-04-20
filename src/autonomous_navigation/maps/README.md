Base map files for autonomous_navigation.

Current setup:
1. base_map.pgm is generated from map_100_bw.png (recommended for precision).
2. base_map.yaml starts with resolution 0.01 m/px as initial calibration.

Run:
1. cd /home/fedir/ROB
2. source /opt/ros/jazzy/setup.bash
3. colcon build --packages-select autonomous_navigation
4. source install/setup.bash
5. ros2 launch autonomous_navigation load_base_map.launch.py

RViz calibration quick loop:
1. Add Map display for /base_map and another for /map.
2. Set alpha around 0.5 and compare wall overlap.
3. Tune base_map.yaml:
   - resolution: scale mismatch (bigger/smaller map)
   - origin[0], origin[1]: XY shift mismatch
   - origin[2]: rotation mismatch (radians)
4. Rebuild and relaunch after YAML changes.

Notes:
- Topic published: /base_map
- Frame used: map
- Planner is configured to prefer /base_map for bounds/occupancy when available.
