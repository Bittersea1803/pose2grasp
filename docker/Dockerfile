FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential libudev-dev usbutils libcanberra-gtk-module libcanberra-gtk3-module openjdk-8-jdk \
    freeglut3-dev libusb-1.0-0-dev g++ curl \
    protobuf-compiler libprotoc-dev python3-opencv ca-certificates python3-dev git wget sudo \
    cmake ninja-build

ENV NO_AT_BRIDGE=1

RUN ln -sv /usr/bin/python3 /usr/bin/python && python --version

RUN wget https://bootstrap.pypa.io/pip/get-pip.py && \
    python3 get-pip.py && rm get-pip.py

RUN pip install tensorboard cmake onnx==1.4.1 tqdm
RUN pip install torch==1.8.1 torchvision===0.9.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6
RUN apt-get install -y python3-tk
RUN pip install numpy matplotlib opencv-python scipy scikit-image tqdm

RUN apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install -y curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt update && apt install -y ros-noetic-desktop
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN apt install -y python3-rosdep && rosdep init && rosdep update

RUN apt-get install -y ros-noetic-openni-launch ros-noetic-openni2-launch ros-noetic-ros-numpy ros-noetic-rosbash

# Postavljamo PYTHONPATH i radni direktorij prema novoj strukturi
ENV PYTHONPATH="${PYTHONPATH}:/home/openpose_user/src/pose2grasp"
WORKDIR /home/openpose_user/src/pose2grasp

CMD ["bash"]