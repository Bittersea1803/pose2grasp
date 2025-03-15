#!/bin/bash

# Omogući X11 pristup (ako već nije, npr. 'xhost +local:root')
xhost +local:root 2>/dev/null

docker rm pose2grasp_docker 2>/dev/null
docker run --gpus all --runtime=runc -it --shm-size=10gb \
    --env="DISPLAY" \
    --volume="/home/marijan/Desktop/pose2grasp:/home/openpose_user/src/pose2grasp" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --workdir="/home/openpose_user/src/pose2grasp" \
    -p 6017:6017 \
    --privileged \
    --name=pose2grasp_docker \
    pose2grasp_docker:latest