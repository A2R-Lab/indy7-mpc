#!/bin/bash

echo -e "----------------------------------------"
echo -e "Starting container...\n"
# docker compose --project-name=mpc up -d 
if ! docker image inspect indy7-mpc &>/dev/null; then
    echo "Building Docker image..."
    docker build -t indy7-mpc .
fi


echo -e "----------------------------------------"
echo -e "Entering container...\n"
# docker compose exec -w /workspace dev bash

export DISPLAY=:0
xhost +local:docker
docker run -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    --network host \
    --rm \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    mpc


# ------ simple instructions
# docker build -t indy7-mpc .

# docker run --gpus all -it -v $(pwd):/workspace 