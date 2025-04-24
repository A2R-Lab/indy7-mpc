#!/bin/bash

# Source common definitions
source "$(dirname "$0")/../../../tools/common.sh"

source /opt/ros/humble/setup.bash

# defaults
TIMESTEP="0.01"
VISUALIZE="false"

# parse options
TEMP=$(getopt -o t:v --long timestep:,visualize -- "$@")
if [ $? != 0 ]; then
    echo -e "${CROSS} ${RED}Terminating...${RESET}" >&2
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -t | --timestep ) TIMESTEP="$2"; shift 2 ;;
        -v | --visualize ) VISUALIZE="true"; shift ;; # Treat presence of -v as true
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo -e "${YELLOW}${BOLD}${GEAR} Project not built. Building now...${RESET}"
    ./make-viz.sh
    source install/setup.bash
fi

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Starting MuJoCo simulator...${RESET}\n"
printf "    ${ARROW} ${BOLD}Sim-timestep:${RESET} ${YELLOW}${TIMESTEP}${RESET}s\n"
if [ "$VISUALIZE" = "true" ]; then
    printf "    ${ARROW} ${BOLD}Visualize:${RESET} ${EYE} ${GREEN}ENABLED${RESET}\n"
else
    printf "    ${ARROW} ${BOLD}Visualize:${RESET} ${RED}DISABLED${RESET}\n"
fi
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"

ros2 run mujoco_sim sim_node $(pwd)/description/indy7.xml "${TIMESTEP}" "${VISUALIZE}"