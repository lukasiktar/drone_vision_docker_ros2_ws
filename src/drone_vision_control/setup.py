from setuptools import find_packages, setup

package_name = 'drone_vision_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['drone_vision_control/Detection.py']),
        ('share/' + package_name, ['drone_vision_control/Recognition.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='istrazivac6',
    maintainer_email='luka.siktar11@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_vision_control=drone_vision_control.drone_vision_control:main' 
        ],
    },
)
