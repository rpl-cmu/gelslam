from setuptools import find_packages, setup

package_name = "gelslam"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/saved_video_launch.py"]),
        ("share/" + package_name + "/launch", ["launch/gelsight_launch.py"]),
        ("share/" + package_name + "/config", ["config/config.yaml"]),
        (
            "share/" + package_name + "/resources/example_calibration/model",
            ["resources/example_calibration/model/nnmodel.pth"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rpl",
    maintainer_email="hungjuih@andrew.cmu.edu",
    description="Gelslam Algorithm",
    license="MIT Liscence",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "saved_video_streamer_node = gelslam.nodes.saved_video_streamer_node:main",
            "gelsight_streamer_node = gelslam.nodes.gelsight_streamer_node:main",
            "tracker_node = gelslam.nodes.tracker_node:main",
            "pose_graph_node = gelslam.nodes.pose_graph_node:main",
            "visualizer_node = gelslam.nodes.visualizer_node:main",
            "gelslam_offline = gelslam.scripts.gelslam_offline:main",
            "gelslam_reconstruct = gelslam.scripts.reconstruct:reconstruct",
            "visualize_pose_graph = gelslam.scripts.visualize_pose_graph:main",
        ],
    },
)
