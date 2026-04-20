from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    map_yaml_default = PathJoinSubstitution(
        [FindPackageShare("autonomous_navigation"), "maps", "base_map.yaml"]
    )

    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value=map_yaml_default,
        description="Absolute path to the occupancy map YAML file",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )
    autostart_arg = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically configure and activate map server lifecycle node",
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="base_map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "yaml_filename": LaunchConfiguration("map_yaml"),
                "topic_name": "/base_map",
                "frame_id": "map",
            }
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_base_map",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "autostart": LaunchConfiguration("autostart"),
                "node_names": ["base_map_server"],
            }
        ],
    )

    return LaunchDescription([
        map_yaml_arg,
        use_sim_time_arg,
        autostart_arg,
        map_server,
        lifecycle_manager,
    ])
