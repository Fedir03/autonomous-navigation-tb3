Base map files for autonomous_navigation.

How to use:
1. Replace base_map.pgm with your real occupancy map image (PGM recommended).
2. Update base_map.yaml with correct resolution and origin.
3. Launch map server:
   ros2 launch autonomous_navigation load_base_map.launch.py

Notes:
- Topic published: /base_map
- Frame used: map
- Planner is configured to prefer base_map for bounds/occupancy when available.
