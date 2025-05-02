#!/bin/bash

# Source common definitions
source "$(dirname "$0")/common.sh"

source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select mujoco_sim

source install/setup.bash

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${CHECK} Build completed successfully.${RESET}\n\n"
printf "${ARROW} ${YELLOW}${BOLD}To source the workspace:${RESET}\n"
printf "  ${BOLD}\$ source install/setup.bash${RESET}\n\n"
printf "${ARROW} ${YELLOW}${BOLD}To run the simulator:${RESET}\n"
printf "  ${BOLD}\$ ./tools/sim.sh${RESET}\n"
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"
