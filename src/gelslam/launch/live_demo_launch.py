import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    data_dir_arg = DeclareLaunchArgument(
        "data_dir", default_value="", description="Directory to save the data"
    )
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=os.path.join(
            get_package_share_directory("gelslam"), "config", "config.yaml"
        ),
        description="Path of the sensor configuration file",
    )
    rendering_arg = DeclareLaunchArgument(
        "rendering",
        default_value="true",
        description="Render the reconstruction process or not",
    )
    streamer_arg = DeclareLaunchArgument(
        "streamer",
        default_value="ffmpeg",
        description="Use ffmpeg or cv2 to stream the sensor readings",
    )
    skip_background_check_arg = DeclareLaunchArgument(
        "skip_background_check",
        default_value="false",
        description="Skip the background check",
    )
    save_gelslam_states_arg = DeclareLaunchArgument(
        "save_gelslam_states",
        default_value="false",
        description="Save the gelslam states or not",
    )

    tracker_node = Node(
        package="gelslam",
        executable="tracker_node",
        name="tracker_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "data_dir": LaunchConfiguration("data_dir"),
                "rendering": LaunchConfiguration("rendering"),
                "save_gelslam_states": LaunchConfiguration("save_gelslam_states"),
                "skip_background_check": LaunchConfiguration("skip_background_check"),
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
                "rendering": LaunchConfiguration("rendering"),
                "save_gelslam_states": LaunchConfiguration("save_gelslam_states"),
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
        condition=IfCondition(LaunchConfiguration("rendering")),
    )
    gelsight_streamer_node = Node(
        package="gelslam",
        executable="gelsight_streamer_node",
        name="gelsight_streamer_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "data_dir": LaunchConfiguration("data_dir"),
                "streamer": LaunchConfiguration("streamer"),
            }
        ],
    )

    return LaunchDescription(
        [
            data_dir_arg,
            config_path_arg,
            rendering_arg,
            streamer_arg,
            skip_background_check_arg,
            save_gelslam_states_arg,
            tracker_node,
            pose_graph_node,
            visualizer_node,
            gelsight_streamer_node,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=tracker_node,
                    on_exit=[EmitEvent(event=Shutdown())],
                )
            ),
        ]
    )
