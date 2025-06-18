from setuptools import setup
import os
from glob import glob

package_name = 'aruco_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gabriel Torres',
    maintainer_email='gabrieltlt721@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    scripts=[
        os.path.join('aruco_detector', 'scripts', 'aruco_detector.py')
    ],
)