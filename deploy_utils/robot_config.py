"""Robot configuration for real deployment. Edit the values below for your setup."""

from pathlib import Path

from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


def create_real_robot() -> Robot:
    """Create and configure a real robot with the specified camera.
    Returns:
        Configured Robot instance 
    """
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",  # CHANGE THIS: your robot's serial port
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path="/dev/video0",  # CHANGE THIS: your webcam device path
            fps=30,
            width=640,
            height=480
        )},
        # cameras={"base_camera": RealSenseCameraConfig(
        #     serial_number_or_name="053645021390",  
        #     fps=30,
        #     width=640,
        #     height=480
        # )},
        id="so101_follower", # CHANGE THIS: your calibration file name
        calibration_dir=Path(__file__).parent,  # CHANGE THIS: path to calibration file directory
    )

    return make_robot_from_config(robot_config)
