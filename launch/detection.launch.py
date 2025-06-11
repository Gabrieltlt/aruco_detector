import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Nome do seu pacote
    pkg_aruco_detector = get_package_share_directory('aruco_detector')

    # <--- MUDANÇA 1: Definir o caminho para o arquivo de configuração YAML
    aruco_config_filepath = os.path.join(pkg_aruco_detector, 'config', 'aruco_config.yaml')

    declared_arguments = [
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true", # Mudei para 'true' para facilitar os testes
            description="Start RViz with a predefined configuration.",
        )
    ]
    use_rviz = LaunchConfiguration("use_rviz")

    # 1. Launch da câmera USB
    launch_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('usb_cam'),
                'launch', 'camera.launch.py'
            )
        )
    )

    # <--- MUDANÇA 2: Modificar o nó do Aruco Detector
    aruco_node = Node(
        package='aruco_detector',
        # Usar o nome do executável definido no 'entry_points' do setup.py
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
        emulate_tty=True,
        parameters=[aruco_config_filepath]
    )

    # 3. RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(pkg_aruco_detector, 'config', 'aruco.rviz')],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        declared_arguments + [
            launch_camera,
            aruco_node,
            rviz_node
        ]
    )