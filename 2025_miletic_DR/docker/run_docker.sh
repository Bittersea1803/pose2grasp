# RUN DOCKER
docker stop miletic_dr

docker rm miletic_dr

docker run --ipc=host --gpus all --runtime=runc --interactive -it \
--shm-size=10gb \
--env="DISPLAY" \
--volume="$(dirname "${PWD}"):/home/RVLuser" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/dev:/dev" \
--workdir="/home/RVLuser" \
--privileged \
--net=host \
--name=miletic_dr miletic_dr:latest

# docker exec -it miletic_dr bash