import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    data_dir_arg = DeclareLaunchArgument(
        "data_dir", default_value="", description="Directory to the saved data"
    )
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=os.path.join(
            get_package_share_directory("gelslam"), "config", "config.yaml"
        ),
        description="Path of the sensor configuration file",
    )
    online_rendering_arg = DeclareLaunchArgument(
        "online_rendering",
        default_value="true",
        description="Online render the reconstruction process or not",
    )
    unbuffered_env = SetEnvironmentVariable("PYTHONUNBUFFERED", "1")

    tracker_node = Node(
        package="gelslam",
        executable="tracker_node",
        name="tracker_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "data_dir": LaunchConfiguration("data_dir"),
            }
        ],
    )
    pose_graph_node = Node(
        package="gelslam",
        executable="pose_graph_node",
        name="pose_graph_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "data_dir": LaunchConfiguration("data_dir"),
                "online_rendering": LaunchConfiguration("online_rendering"),
            }
        ],
    )
    visualizer_node = Node(
        package="gelslam",
        executable="visualizer_node",
        name="visualizer_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "data_dir": LaunchConfiguration("data_dir"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("online_rendering")),
    )
    saved_video_streamer_node = Node(
        package="gelslam",
        executable="saved_video_streamer_node",
        name="saved_video_streamer_node",
        output="screen",
        parameters=[{"data_dir": LaunchConfiguration("data_dir")}],
    )

    return LaunchDescription(
        [
            data_dir_arg,
            config_path_arg,
            online_rendering_arg,
            tracker_node,
            pose_graph_node,
            visualizer_node,
            saved_video_streamer_node,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=saved_video_streamer_node,
                    on_exit=[EmitEvent(event=Shutdown())],
                )
            ),
        ]
    )
