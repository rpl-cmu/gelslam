<h1 align="center">
    GelSLAM: A Real-Time, High-Fidelity, and Robust <br/> 3D Tactile SLAM System
</h1>

<div align="center">
  <a href="https://joehjhuang.github.io/" target="_blank">Hung-Jui Huang</a> &nbsp;•&nbsp;
  <a href="https://www.aminmirzaee.com/" target="_blank">Mohammad Amin Mirzaee</a> &nbsp;•&nbsp;
  <a href="https://www.cs.cmu.edu/~kaess/" target="_blank">Michael Kaess</a> &nbsp;•&nbsp;
  <a href="https://siebelschool.illinois.edu/about/people/all-faculty/yuanwz" target="_blank">Wenzhen Yuan</a>
</div>

<h4 align="center">
  <a href="https://joehjhuang.github.io/gelslam"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Web.svg" alt="Website" width="10px"/> <b>Website</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/pdf/2508.15990"><img src="assets/arxiv.png" alt="arXiv" width="28px"/> <b>Paper</b></a> &nbsp;&nbsp;&nbsp; &nbsp;
  🤗 <a href="https://huggingface.co/datasets/joehjhuang/GelSLAM"> <b>Dataset</b></a>
</h4>


<div align="center">
<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp; 
<a href="https://rpl.ri.cmu.edu/" target="_blank">
  <img height="20" src="assets/rpl.png" alt="RPL">
</a>
</div>

<p align="center">
  <img src="assets/teaser.gif" alt="GelSLAM">
</p>

GelSLAM is a real-time 3D SLAM system that enables ultra-high-fidelity object shape reconstruction and precise object pose tracking relying on tactile sensing alone. It accomplishes what was previously out of reach for tactile-only systems. For more details and results, please visit our [website](https://joehjhuang.github.io/gelslam) and the [arXiv paper](https://arxiv.org/abs/2508.15990). 



## System Requirements
The codebase has been tested with the following configuration. While other versions may work, they have not been verified.

- **OS**: Ubuntu 22.04
- **Python**: 3.10.12
- **ROS**: ROS 2 Humble (Desktop Full)
  - We recommend a system-wide installation for all components (ROS 2, GTSAM, and Python dependencies) because ROS 2 is difficult to configure within virtual environments. Usage with virtual environments is not verified.
- **GTSAM**: 4.3.0
  - Must be compiled with Python bindings enabled (`-DGTSAM_BUILD_PYTHON=1`).
  - Refer to the [GTSAM Python Installation Guide](https://github.com/borglab/gtsam/tree/develop/python) for detailed instructions.

## Installation
1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Build GelSLAM**:
    ```bash
    colcon build
    source install/setup.bash
    ```

## Quick Start

Experience GelSLAM’s real-time reconstruction by replaying our sample GelSight video of scanning a seed. The video streams tactile images at the sensor's native frame rate to simulate live hardware input. To run a live demo on your own GelSight Mini sensor or reproduce the paper reconstruction results, see [Usage](#usage).

1.  **Download Sample Data**:
    ```bash
    bash download_sample_data.sh
    ```

2.  **Run Real-Time Reconstruction**:
    ```bash
    ros2 launch gelslam saved_video_launch.py data_dir:=data/sample_data/example_scan config_path:=data/sample_data/example_scan/config.yaml
    ```

A window will open displaying the live reconstruction. After playback finishes, the final mesh is saved to `data/sample_data/example_scan/gelslam_online/reconstructed_mesh.ply`.

## Usage
GelSLAM reconstruction supports three operating modes: **Live** (real-time sensor input), **Live Replay** (real-time processing of recorded video), and **Offline** (offline processing of recorded video). 

### 0. Prerequisites
All three operating modes require a sensor calibration and a configuration file.

- **Sensor Calibration**:  
  Use the [GelSight SDK](https://github.com/joehjhuang/gs_sdk) to calibrate your sensor. <span style="color:red">A new calibration is required if the GelPad is replaced or if a different sensor is used.</span> Calibration generally takes less than an hour. An example calibration directory is provided at `src/gelslam/resources/example_calibration/`, with the calibration model located at `src/gelslam/resources/example_calibration/model/nnmodel.pth`.
- **Configuration File**:  
  GelSLAM is configured via `src/gelslam/config/config.yaml`. Copy this file to your data directory and update the `calibration_model_path` to point to your specific sensor calibration model. You typically don't need to modify other parameters.

### 1. Live Reconstruction
Run real-time reconstruction directly on your own GelSight Mini sensor.
```bash
ros2 launch gelslam gelsight_launch.py data_dir:=<path_to_save_data> config_path:=<path_to_config>
```
- **Arguments**

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Directory where the recorded video and reconstructed mesh will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `online_rendering` | No | Enable rendering of reconstruction progress (default: `true`) |
  | `streamer` | No | ffmpeg or cv2, please see notes (default: `ffmpeg`) |
  | `skip_background_check` | No | Skip background image confirmation, please see notes (default: `false`) |
  | `save_gelslam_states` | No | Save the GelSLAM states (default: `false`) |

- **Instructions**
  - <span style="color:red">After launch, wait about 3 seconds before scanning to allow background capture.</span> The captured background image is shown for confirmation to prevent accidental contact generating an incorrect background. Use `skip_background_check` to skip this confirmation.
  - Press **Ctrl+C** to stop scanning at any time.
  - <span style="color:red">By default, GelSLAM uses an FFmpeg-based streamer at 25 Hz for GelSight Mini. On some systems, this may cause frame delay or duplication after 20 seconds of scanning. If this occurs, set `streamer` to `cv2`. However, this reduces the frame rate to approximately 10 Hz, so scan slower with this setting.</span>
  - Online rendering is not part of the GelSLAM algorithm. As the mesh grows, rendering can slow down the system and affect performance during long scans. Rendering resolution is automatically reduced as the mesh grows, while GelSLAM continues to run at full resolution.
  - Avoid continuous live scanning longer than 10 minutes, even with `online_rendering` disabled. For long scans, record multiple segments, stitch the GelSight videos, and use offline reconstruction.

- **Output**  
  After stopping the scan, GelSLAM fuses and saves the reconstructed mesh to: `<data_dir>/gelslam_online/reconstructed_mesh.ply`. For long scans, mesh fusion and saving may take up to a minute.
    
### 2. Live Replay Reconstruction
Run real-time reconstruction by replaying a recorded GelSight video. Tactile images are streamed at the sensor’s native frame rate to simulate live input. This is the mode used in **Quick Start**. Compared to Live Reconstruction, the only difference is the source of the tactile image stream. Arguments and usage are otherwise similar.
```bash
ros2 launch gelslam saved_video_launch.py data_dir:=<path_to_recorded_data> config_path:=<path_to_config>
```
- **Arguments**

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Directory containing the recorded GelSight video and where reconstructed mesh will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `online_rendering` | No | Enable rendering of reconstruction progress (default: `true`) |
  | `save_gelslam_states` | No | Save the GelSLAM states (default: `false`) |

- **Instructions**
  - Place the GelSight video in `data_dir` and name it `gelsight.avi`. <span style="color:red">Ensure no contact occurs during the first 10 frames, which are used as GelSight background.</span>
  - Once the video stream ended, the process will terminate itself. Otherwise, you may press **Ctrl+C** to stop at any time.
  - Online rendering is not part of the GelSLAM algorithm. As the mesh grows, rendering can slow down the system and affect performance during long scans. 
  - Avoid continuous live reconstruction longer than 10 minutes, even with `online_rendering` disabled. For long scans, use offline reconstruction.

- **Output**  
  After completion (or early termination), GelSLAM fuses and saves the reconstructed mesh to: `<data_dir>/gelslam_online/reconstructed_mesh.ply`. For long videos, mesh fusion and saving may take up to a minute.

### 3. Offline Reconstruction
Run offline reconstruction on a recorded GelSight video. This mode maximizes reconstruction quality and is recommended for long scans (over 10 minutes).

```bash
ros2 run gelslam gelslam_offline -d <path_to_data> -c <path_to_config>
```
- **Arguments**

  | Argument | Required | Description |
  |---|:---:|---|
  | `data_dir` | **Yes** | Directory containing the recorded GelSight video and where reconstructed mesh will be saved |
  | `config_path` | **Yes** | Path to the GelSLAM configuration file |
  | `rendering` | No | Enable rendering of reconstruction progress (default: `false`) |
  | `save_gelslam_states` | No | Save the GelSLAM states (default: `false`) |

- **Instructions**
  - Place the GelSight video in `data_dir` and name it `gelsight.avi`. <span style="color:red">Ensure no contact occurs during the first 10 frames, which are used as GelSight background.</span>
  - The process terminates automatically once the reconstruction ends.
  - Rendering may be enabled to visualize reconstruction progress, but we recommend keeping it disabled to maximize reconstruction speed.
  - For extremely long videos (30 minutes), memory usage may become a concern, as the recorded video itself can be several gigabytes.

- **Output**  
  After completion, the reconstructed mesh is saved to: `<data_dir>/gelslam_offline/reconstructed_mesh.ply`.

### Notes:

- Rendering resolution is automatically reduced as the mesh grows, while GelSLAM continues to run at full resolution.
---

## Examples
This example demonstrates basic usage of NormalFlow. Run the command below to test the tracking algorithm.
```bash
test_tracking [-d {cpu|cuda}]
```
The command reads the tactile video `examples/data/tactile_video.avi`, tracks the touched object, and saves the result in `examples/data/tracked_tactile_video.avi`.

## Documentation
The `normalflow` function in `normalflow/registration.py` implements frame-to-frame NormalFlow tracking, returning the homogeneous transformation from a reference sensor frame to a target sensor frame (see figure below). If tracking fails, it raises `InsufficientOverlapError`. For usage, see `examples/test_tracking.py` and `demos/realtime_object_tracking.py`.

<p align="center">
  <br>
  <img src="assets/coordinate_conventions.png" alt="Coordinate Conventions" style="width:40%; height:auto;">
  <br>
</p>

## Reproduce Paper Results
To reproduce the main results from our [paper](https://ieeexplore.ieee.org/document/10766628), which compares NormalFlow with baseline algorithms, please visit the [NormalFlow Experiment](https://github.com/rpl-cmu/normalflow_experiment) repository.

## Updates
* **2025-02-03**: Implemented the NormalFlow failure detection method using Curvature Cosine Similarity (CCS) and Shared Contact Ratio (SCR) metrics. Additionally, integrated the subsampling strategy that prioritizes high-curvature regions instead of random subsampling. Both improvements are based on the [GelSLAM paper](https://joehjhuang.github.io/gelslam/).

## Cite NormalFlow
If you find this package useful, please consider citing our paper:
```
@ARTICLE{huang2024normalflow,
    author={Huang, Hung-Jui and Kaess, Michael and Yuan, Wenzhen},
    journal={IEEE Robotics and Automation Letters}, 
    title={NormalFlow: Fast, Robust, and Accurate Contact-based Object 6DoF Pose Tracking with Vision-based Tactile Sensors}, 
    year={2024},
    volume={},
    number={},
    pages={1-8},
    keywords={Force and Tactile Sensing, 6DoF Object Tracking, Surface Reconstruction, Perception for Grasping and Manipulation},
    doi={10.1109/LRA.2024.3505815}}
```
