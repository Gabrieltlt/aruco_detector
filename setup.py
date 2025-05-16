from setuptools import setup
import os
from glob import glob

package_name = 'aruco_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],  # Vazio pois não usaremos pacote Python tradicional
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),  # Se tiver configs
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    scripts=[
        os.path.join('aruco_detector', 'scripts', 'aruco_detector.py')
    ],
)