#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

# Se você usar execução remota, descomente a linha abaixo e adicione o pacote launch_remote_ssh ao seu ambiente
# from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote

def generate_launch_description():

    config_file_path = PathJoinSubstitution([
        FindPackageShare('aruco_detector'),
        'config',
        'aruco_recognition.yaml']
    )

    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'config',
            default_value=config_file_path,
            description='Path to the parameter file'
        ))

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_remote',
            default_value='false',
            description="If it should run the node on remote"
        ))
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description="Start RViz with a predefined configuration."
        ))

    aruco_recognition_node = Node(
        package='aruco_detector',
        executable='aruco_recognition',
        name='aruco_recognition',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('config')],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    # Camera USB (ajuste o nome do pacote/launch se necessário)
    usb_cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('usb_cam'),
                'launch', 'camera.launch.py'
            )
        )
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare('aruco_detector'), 'config', 'aruco.rviz'])],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        *declared_arguments,
        aruco_recognition_node,
        usb_cam_node,
        rviz_node
    ])