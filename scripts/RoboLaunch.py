#!/usr/bin/env python3
"""Full AnyGrasp-driven bringup that replaces keyboard control with tracking+grasping."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command
from launch.substitutions import FindExecutable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _scripts_dir() -> str:
    pkg_share = get_package_share_directory('open_manipulator_bringup')
    return os.path.normpath(os.path.join(pkg_share, '..', '..', 'scripts'))


def generate_launch_description():
    scripts_dir = _scripts_dir()
    tracking_path = os.path.join(scripts_dir, 'tracking.py')
    grasping_path = os.path.join(scripts_dir, 'grasping.py')

    declared_arguments = [
        DeclareLaunchArgument('start_rviz', default_value='false'),
        DeclareLaunchArgument('prefix', default_value='""'),
        DeclareLaunchArgument('use_sim', default_value='false'),
        DeclareLaunchArgument('use_mock_hardware', default_value='false'),
        DeclareLaunchArgument('mock_sensor_commands', default_value='false'),
        DeclareLaunchArgument('port_name', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('init_position', default_value='true'),
        DeclareLaunchArgument(
            'ros2_control_type', default_value='open_manipulator_x_position'
        ),
        DeclareLaunchArgument(
            'init_position_file', default_value='initial_positions.yaml'
        ),
        DeclareLaunchArgument('start_grasping', default_value='true'),
        DeclareLaunchArgument('start_static_tf', default_value='true'),
    ]

    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_sim = LaunchConfiguration('use_sim')
    use_mock_hardware = LaunchConfiguration('use_mock_hardware')
    mock_sensor_commands = LaunchConfiguration('mock_sensor_commands')
    port_name = LaunchConfiguration('port_name')
    init_position = LaunchConfiguration('init_position')
    ros2_control_type = LaunchConfiguration('ros2_control_type')
    init_position_file = LaunchConfiguration('init_position_file')
    start_grasping = LaunchConfiguration('start_grasping')
    start_static_tf = LaunchConfiguration('start_static_tf')

    urdf_file = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('open_manipulator_description'),
            'urdf',
            'open_manipulator_x',
            'open_manipulator_x.urdf.xacro',
        ]),
        ' ',
        'prefix:=',
        prefix,
        ' ',
        'use_sim:=',
        use_sim,
        ' ',
        'use_mock_hardware:=',
        use_mock_hardware,
        ' ',
        'mock_sensor_commands:=',
        mock_sensor_commands,
        ' ',
        'port_name:=',
        port_name,
        ' ',
        'ros2_control_type:=',
        ros2_control_type,
    ])

    controller_manager_config = PathJoinSubstitution([
        FindPackageShare('open_manipulator_bringup'),
        'config',
        'open_manipulator_x',
        'hardware_controller_manager.yaml',
    ])

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('open_manipulator_description'),
        'rviz',
        'open_manipulator.rviz',
    ])

    trajectory_params_file = PathJoinSubstitution([
        FindPackageShare('open_manipulator_bringup'),
        'config',
        'open_manipulator_x',
        init_position_file,
    ])

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': urdf_file}, controller_manager_config],
        output='both',
        condition=UnlessCondition(use_sim),
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', 'gripper_controller', 'joint_state_broadcaster'],
        parameters=[{'robot_description': urdf_file}],
        output='both',
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_file, 'use_sim_time': use_sim}],
        output='both',
    )

    joint_trajectory_executor = Node(
        package='open_manipulator_bringup',
        executable='joint_trajectory_executor',
        parameters=[trajectory_params_file],
        output='both',
        condition=IfCondition(init_position),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='both',
        condition=IfCondition(start_rviz),
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.2', '0.0', '0.25', '0', '0', '0', 'world', 'camera_link'],
        condition=IfCondition(start_static_tf),
        output='screen',
    )

    grasping_process = ExecuteProcess(
        cmd=['python3', grasping_path],
        output='screen',
        condition=IfCondition(start_grasping),
    )

    grasping_after_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[grasping_process],
        )
    )

    init_positions_after_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[joint_trajectory_executor],
        )
    )

    rviz_after_controllers = RegisterEventHandler(
        OnProcessExit(target_action=robot_controller_spawner, on_exit=[rviz_node])
    )

    return LaunchDescription(
        declared_arguments
        + [
            control_node,
            robot_controller_spawner,
            robot_state_publisher_node,
            static_tf_node,
            rviz_after_controllers,
            init_positions_after_controllers,
            grasping_after_controllers,
        ]
    )
