import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Get the package share directory
    sim_package_share = get_package_share_directory('sim')

    # Declare launch arguments
    model_file_arg = DeclareLaunchArgument(
        'model_file',
        default_value=os.path.join(sim_package_share, '../../description', 'indy7.xml'),
        description='Path to the MuJoCo MJCF model file.'
    )
    timestep_arg = DeclareLaunchArgument(
        'timestep',
        default_value='0.01',
        description='Simulation timestep.'
    )
    visualize_arg = DeclareLaunchArgument(
        'visualize',
        default_value='true',
        description='Enable visualization.'
    )

    sim_node = Node(
        package='sim',
        executable='sim_node',
        name='sim_node', 
        arguments=[
            LaunchConfiguration('model_file'),
            LaunchConfiguration('timestep'),
            LaunchConfiguration('visualize')
        ],
        output='screen'
    )

    return LaunchDescription([
        model_file_arg,
        timestep_arg,
        visualize_arg,
        sim_node
    ]) 