#!/bin/bash

source /opt/ros/humble/setup.bash

if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo "Project not built. Building now..."
    ./make-viz.sh
    source install/setup.bash
fi

echo "--------------------------------"
echo "Running Indy7 MuJoCo simulator..."
echo "--------------------------------"
ros2 launch sim sim.launch.py