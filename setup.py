import os
import glob
from setuptools import find_packages, setup

package_name = 'aruco_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        # (os.path.join('share', package_name, 'config/yolo_tracker_config'), glob.glob('config/yolo_tracker_config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gabriel Torres',
    maintainer_email='gabrieltlt721@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reference_node = aruco_detector.reference_folder.reference_node:main',
            'aruco_recognition = aruco_detector.aruco_recognition.aruco_recognition:main',
        ],
    },
)