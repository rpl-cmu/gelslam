# Requirements
* Ubuntu 22.04
* Python 3.10.12
* pip Packages:
  * Pytorch>=2.5.0
  * Open3D==0.18.0
  * numpy>=1.26.4
  * cv2==4.11.0
  * scipy==1.15.2
* ROS 2 humble (make sure to install development tools as well)
  * https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
* [GTSAM 4.3.0]( with [Python wrapper] (make sure you install it for Python 3.10.12 `cmake -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_PYTHON-1 -DGTSAM_PYTHON_VERSION=3.10.12`)
  * https://github.com/borglab/gtsam
  * https://github.com/borglab/gtsam/tree/develop/python
* Install gs_sdk and normalflow packages.
* git clone the current repo and `colcon build`
* Source the ros_humble and the workspace, then:
* `ros2 launch touch_slam gelsight_launch.py parent_dir:=/your/saving/directory use_profiler:=false sigterm_timeout:=20 config_path:=/your/config/path calib_dir:=/your/calib/dir`, as default, use the config path: `touch_slam/touch_slam/configs/small_objects.yaml`
* Replay the video and reconstruct in real-time [multi_threaded]
`ros2 launch touch_slam saved_video_launch.py parent_dir:=/your/saving/directory config_path:=/your/config/path calib_dir:=/your/calib/dir use_profiler:=false`
* Reconstruct offline [single_threaded] (run it in the directory your_ws/src/touch_slam/touch_slam)
`python -m touch_construct.main_no_visualizer -p /your/saving/directory -b /your/calib/dir -c /your/config/path`
* Continue Reconstruct offline [single_threaded] (run it in the directory your_ws/src/touch_slam/touch_slam), if you have a reconstructed directory (init_dir) and want to continue reconstruct based on another directory (parent_dir), run this
`python -m touch_construct.main_no_visualizer -p /your/saving/directory -b /your/calib/dir -c /your/config/path -i /your/init/directory`
* Tune the surface_info_config: Sometimes, the reconstruction result is bad because the algorithm includes a lot of non-contacted regions as part of the object. You will need to tune the `surface_info_config/height_threshold` parameter in the configuration file. After changing the configuration, run this to generate a contact mask video `contact_masks.avi` to manually check if the mask is good: `python -m touch_construct.tuning.viz_contact_masks -p /your/saving/directory -c /your/config/path -b /your/calib/dir`

* After any of the above methods:
`python -m touch_construct.postprocessing.reconstruct -p /your/saving/directory -c /your/config/path -m [single_threaded or multi_threaded]`

* Finally, you can run this if you believe there are outliers in loop closure:
`python -m touch_construct.postprocessing.robust_reconstruct -p /your/saving/directory -c /your/config/path -m single_threaded`

* To check the Pose Graph, run the following:
`python -m touch_construct.postprocessing.plot_pose_graph -p /your/saving/directory -c /your/config/path -m [single_threaded or multi_threaded or robust_single_threaded]`, if you use robust reconstruction, run with `-m robust_single_threaded`


# TODOs:
* (Optional) Online Robust Solver
* Refactor the data saving module. For example, saving PoseGraph, KeyframeDB, ...
