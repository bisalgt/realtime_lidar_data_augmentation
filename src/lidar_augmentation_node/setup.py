from setuptools import setup

package_name = 'lidar_augmentation_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bisalgt',
    maintainer_email='bisalgt@gmail.com',
    description='lidar_augmentation_node_description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'listen_modify_publish = lidar_augmentation_node.lidar_data_augmentation_function:main',
        ],
    },
)
