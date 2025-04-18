#!/bin/bash

source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select sim  --cmake-args -DENABLE_VISUALIZATION=ON

source install/setup.bash

echo "Build completed for sim (Viz enabled). Source the workspace with:"
echo ""
echo " > source install/setup.bash"
echo ""
