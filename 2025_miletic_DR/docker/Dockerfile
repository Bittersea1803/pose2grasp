FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y wget build-essential git unzip cmake g++ python


###### RVL ######
# Preparing for RVL installation
RUN apt install -y libeigen3-dev
RUN apt-get install -y cmake-curses-gui
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev libgl1-mesa-glx xvfb
RUN apt-get update
RUN apt-get install -y gdb libgtk2.0-dev pkg-config
RUN apt-get install -y libhdf5-serial-dev
RUN apt-get update
RUN apt-get install -y libusb-1.0-0-dev libudev-dev
RUN apt-get install -y default-jdk openjdk-11-jdk
RUN apt-get install -y libtiff-dev freeglut3-dev doxygen graphviz

WORKDIR /
# Install VTK 7.1.1
RUN wget https://gitlab.kitware.com/vtk/vtk/-/archive/v7.1.1/vtk-v7.1.1.tar.gz
RUN tar -xf vtk-v7.1.1.tar.gz
RUN apt install -y libgl1-mesa-dev libxt-dev
RUN cd vtk-v7.1.1 && mkdir build && cd build && cmake -DBUILD_TESTING=OFF .. && make -j$(nproc) && make install

# Install OpenCV 3.4.16
RUN wget https://github.com/opencv/opencv/archive/3.4.16.zip
RUN unzip 3.4.16.zip
# Install OpenCV Contrib
RUN git clone --depth 1 --branch '3.4.16' https://github.com/opencv/opencv_contrib.git
RUN ls /opencv_contrib
WORKDIR /opencv-3.4.16
RUN mkdir build && cd build && cmake -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules -DWITH_EIGEN=ON -DWITH_VTK=ON -DBUILD_opencv_world=ON .. && make -j$(nproc) && make install


# Install FLANN 1.8.4
WORKDIR /
RUN git clone --depth 1 --branch '1.8.4' https://github.com/flann-lib/flann.git
RUN cd flann && touch src/cpp/empty.cpp && sed -e '/add_library(flann_cpp SHARED/ s/""/empty.cpp/' \
    -e '/add_library(flann SHARED/ s/""/empty.cpp/' \
    -i src/cpp/CMakeLists.txt
RUN cd flann && mkdir build && cd build && cmake .. && make -j$(nproc) && make install

# Add RVL to the LD_LIBRARY_PATH
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/home/RVLuser/rvl-linux/build/lib" >> /etc/bash.bashrc

RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip3 install numpy

# Install pybind 11 for RVL
RUN pip3 install pybind11
RUN apt-get update

# Add RVL python modules to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/home/RVLuser/rvl-linux/python/build/lib"
ENV PYTHONPATH="${PYTHONPATH}:/home/RVLuser/rvl-linux/modules/RVLPY"
ENV PYTHONPATH="${PYTHONPATH}:/home/RVLuser/rvl-linux/python"

### FCL installation ###
WORKDIR /
# libccd
RUN wget -O libccd-2.1.tar.gz https://github.com/danfis/libccd/archive/refs/tags/v2.1.tar.gz
RUN tar -xvf libccd-2.1.tar.gz
RUN cd libccd-2.1 && mkdir build && cd build && cmake .. -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC" && make -j$(nproc) && make install

# octomap
RUN wget https://github.com/OctoMap/octomap/archive/refs/tags/v1.10.0.tar.gz -O octomap-1.10.0.tar.gz
RUN tar -xvf octomap-1.10.0.tar.gz
RUN cd octomap-1.10.0 && mkdir build && cd build && cmake .. && make -j$(nproc) && make install

# FCL 
RUN wget https://github.com/flexible-collision-library/fcl/archive/refs/tags/0.7.0.tar.gz -O fcl-0.7.0.tar.gz
RUN tar -xvf fcl-0.7.0.tar.gz
# delete line 241 because it errors out - deleting it will fix
RUN sed -i '241d' fcl-0.7.0/CMakeLists.txt 
RUN cd fcl-0.7.0 && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_PREFIX_PATH=/usr/local \
    -DOctomap_DIR=/usr/local/share/octomap \
    && make -j$(nproc) && make install
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# FCL python
RUN pip3 install python-fcl==0.7.0.6

###### ROS-UR5 ######
# Install ROS Noetic
RUN apt-get update
RUN apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install -y curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
RUN apt install -y ros-noetic-desktop-full
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-catkin-tools

# Install UR5 dependencies
RUN apt install python3-rosdep
RUN apt-get install -y ros-noetic-realsense2-camera
RUN apt-get install -y ros-noetic-openni-launch
RUN apt-get install -y ros-noetic-openni2-launch
RUN apt-get install -y ros-noetic-rosbash
RUN apt-get install -y ros-noetic-ros-control
RUN apt-get install -y ros-noetic-soem
RUN apt-get install -y ros-noetic-moveit
RUN apt-get install -y ros-noetic-trac-ik
RUN apt-get install -y ros-noetic-industrial-core ros-noetic-ros-industrial-cmake-boilerplate ros-noetic-socketcan-interface ros-noetic-industrial-robot-status-interface ros-noetic-ros-controllers ros-noetic-scaled-joint-trajectory-controller ros-noetic-speed-scaling-interface ros-noetic-speed-scaling-state-controller ros-noetic-ur-msgs ros-noetic-pass-through-controllers ros-noetic-ur-client-library

# Install UR5 dependencies
RUN pip3 install pymodbus --upgrade
RUN pip3 install ur-rtde
RUN pip3 install pyyaml

# Link python3 to python for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install open3d for point cloud processing and visualization
RUN pip3 install open3d
# Install Pillow for image processing
RUN pip3 install Pillow==9.0.0

# Grasp detection dependencies
RUN pip3 install scikit-image
RUN pip3 install xgboost
RUN pip3 install torch

# Add core package in UR5 to the PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/home/RVLuser/ferit_ur5_ws/src/core/src"