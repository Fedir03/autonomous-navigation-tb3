Base map files for autonomous_navigation.

Current setup:
1. base_map.pgm is generated from map_100_bw.png (recommended for precision).
2. base_map.yaml uses external/professor frame as reference.
3. Runtime alignment to SLAM map is computed from the initial pose entered in autonomous_navigation.

Run:
1. cd /home/fedir/ROB
2. source /opt/ros/jazzy/setup.bash
3. colcon build --packages-select autonomous_navigation
4. source install/setup.bash
5. ros2 launch autonomous_navigation load_base_map.launch.py

Frame notes:
1. /base_map is published in frame base_map_ext by default.
2. autonomous_navigation publishes a runtime TF map -> base_map_ext from the entered initial pose.
3. This allows changing initial pose without re-editing map origin/yaw for each run.

RViz calibration quick loop:
1. Add Map display for /base_map and another for /map.
2. Set alpha around 0.5 and compare wall overlap.
3. Tune base_map.yaml:
   - resolution: scale mismatch (bigger/smaller map)
   - keep origin close to [0, 0, 0] when using runtime initial-pose alignment
4. Rebuild and relaunch after YAML changes.

Notes:
- Topic published: /base_map
- Frame used: map
- Planner is configured to prefer /base_map for bounds/occupancy when available.
