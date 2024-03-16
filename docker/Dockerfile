FROM ubuntu:focal
SHELL ["/bin/bash", "-c"]

# Settings
ENV ARCH=amd64
ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=noetic

# Permissions
ARG UID
ARG GID
ARG USER=docker
ARG PASS=docker
ARG SSH_PRIV_KEY
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PASS}" | chpasswd
RUN apt-get update && apt-get install -y sudo
RUN adduser ${USER} sudo
ENV HOME /home/$USER

# Install deps
RUN apt-get install -qy \
  git \
  tree \
  htop \
  tmux \
  exuberant-ctags \
  automake \
  cmake \
  ccache \
  gcc \
  clang \
  clang-format \
  clang-tidy \
  protobuf-compiler \
  libpython3-dev \
  pylint \
  yapf3

# Install ROS
RUN apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-get install curl gnupg -qy
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-get update
RUN apt-get install ros-${ROS_DISTRO}-desktop -qy

# Install Gazebo-Fortress
RUN apt-get install wget -qy
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install ignition-fortress -qy

# Install MAVROS
RUN apt-get install ros-${ROS_DISTRO}-mavros ros-${ROS_DISTRO}-mavros-extras -qy

# Install additional tools and libraries
RUN apt-get install -qy \
  python3-pip \
  python3-numpy \
  python3-scipy \
  python3-pandas \
  python3-seaborn \
  python3-matplotlib \
  python3-setuptools \
  python3-catkin-tools \
  libeigen3-dev \
  libopencv-dev

# Switch to user
USER ${USER}

# Setup catkin workspace
WORKDIR $HOME
RUN mkdir -p $HOME/catkin_ws/src
RUN cd $HOME/catkin_ws \
  && source /opt/ros/${ROS_DISTRO}/setup.bash \
  && catkin init \
  && catkin build

# Install GeographicLib datasets
WORKDIR $HOME/catkin_ws
USER root
RUN wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
RUN chmod +x install_geographiclib_datasets.sh
RUN sudo bash ./install_geographiclib_datasets.sh
RUN rm install_geographiclib_datasets.sh
USER ${USER}

# Build catkin workspace
WORKDIR $HOME/catkin_ws
USER root
RUN apt-get install \
  libgflags-dev \
  libgoogle-glog-dev \
  ros-${ROS_DISTRO}-gazebo-msgs \
  libignition-gazebo3-* \
  libignition-transport8-* \
  -qy
USER ${USER}
RUN env IGNITION_VERSION=fortress catkin build -DCMAKE_BUILD_TYPE=Release

# Install QGroundControl
USER root
RUN usermod -a -G dialout $USER
RUN apt-get remove modemmanager -y
RUN apt install gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl -y
RUN apt install fuse libfuse2 -y
RUN apt install libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0 -y
USER ${USER}
WORKDIR $HOME
RUN wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage
RUN chmod +x QGroundControl.AppImage

# Tools
USER root
WORKDIR $HOME
# -- Install Neovim
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz \
  && sudo rm -rf /opt/nvim \
  && sudo tar -C /opt -xzf nvim-linux64.tar.gz \
  && rm nvim-linux64.tar.gz \
  && echo 'export PATH=$PATH:/opt/nvim-linux64/bin' > $HOME/.bashrc
# -- Install vim and vifm
RUN apt-get install -qy vim vifm
# -- Install dotfiles
USER ${USER}
RUN git clone https://github.com/chutsu/dotfiles \
  && cd dotfiles \
  && make dotfiles

# Entry point script
WORKDIR $HOME
COPY entry.sh .
ENTRYPOINT ["/home/docker/entry.sh"]
