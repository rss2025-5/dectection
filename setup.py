from setuptools import find_packages, setup
import glob
import os

package_name = 'dectection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/dectection/launch/', glob.glob(os.path.join('launch', 'part_a', '*launch.*'))),
        # (os.path.join('share', package_name, 'config', 'real'), glob.glob('config/real/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='jialinc7@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection = dectection.final_challenge2025.shrinkray_heist.model.detection_node:main',
            'drive = dectection.drive_mux_node:main',
            'state = dectection.state_machine_node:main'

        ],
    },
)
