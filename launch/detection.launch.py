import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    # Argumentos (opcional, para habilitar/desabilitar RViz)
    declared_arguments = [
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Start RViz2 automatically with this launch file.",
        )
    ]
    use_rviz = LaunchConfiguration("use_rviz")

    # 1. Launch da câmera USB (usb_cam)
    launch_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('usb_cam'),
                'launch', 'camera.launch.py'  # Confirme o nome do arquivo!
            )
        )
    )

    # 2. Nó do aruco_detector (ajuste o nome do executável se necessário)
    aruco_node = Node(
        package='aruco_detector',
        executable='aruco_detector.py',  # Verifique com `ros2 pkg executables aruco_detector`
        name='aruco_detector',
        output='screen',
        parameters=[{
            'camera_topic': '/camera1/image_raw',  # Tópico corrigido!
            'debug_mode': False,  # Opcional
            # Adicione outros parâmetros específicos do seu nó aqui
        }]
    )

    # 3. RViz (opcional, para visualização)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(get_package_share_directory('aruco_detector'), 'config', 'aruco.rviz')],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        declared_arguments + [
            launch_camera,
            aruco_node,
            rviz_node  # Remova se não for usar RViz
        ]
    )