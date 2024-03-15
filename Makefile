SSH_PRIV_KEY=`cat ~/.ssh/id_rsa`

help: ## Help
	@echo -e "\033[1;34m[make targets]:\033[0m"
	@egrep -h '\s##\s' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;36m%-20s\033[0m %s\n", $$1, $$2}'

install_docker: ## Install Docker (Ubuntu only)
	@bash scripts/install_docker.bash

build_docker:  ## Build docker container
	cd docker && docker build -t mav_exp .

run_docker: ## Run docker container
	@xhost +local:docker && docker run \
		-e DISPLAY \
		-v $(PWD):/home/docker/catkin_ws/src/mav_exp \
		-v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' \
		-v /dev/shm:/dev/shm \
		-v /dev/dri:/dev/dri \
		--privileged \
		--ipc=host \
		--pid=host \
		--network="host" \
		-it --rm mav_exp /bin/bash

run_docker_mav: ## Run docker container on the MAV
	@docker run \
		--privileged \
		--ipc=host \
		--pid=host \
		--network="host" \
		--device=/dev/ttyACM0 \
		-it --rm mav_exp /bin/bash

vicon_sdk:  ## Install Vicon SDK
	@echo "Downloading Vicon DataStream SDK 1.12"
	@rm -rf 20230413_145507h
	@rm -f 7s2j1j2oec1d19up2p5se5tlqrsky149.zip
	@rm -rf vicon_datastream_sdk
	@wget -q --show-progress https://app.box.com/shared/static/7s2j1j2oec1d19up2p5se5tlqrsky149.zip
	@unzip 7s2j1j2oec1d19up2p5se5tlqrsky149.zip
	@mv 20230413_145507h vicon_datastream_sdk
	@rm 7s2j1j2oec1d19up2p5se5tlqrsky149.zip

catkin_ws:  ## Build catkin workspace
	cd ${HOME}/catkin_ws/src && catkin build

fix_px4: ## Fix PX4 in gazebo mode
	rosrun mavros mavparam set COM_RCL_EXCEPT 4  # RC LOSS EXCEPTION -> 4 (Not documented)
	rosrun mavros mavparam set NAV_DLL_ACT 0     # GCS loss failsafe mode -> 0 (Disabled)
	rosrun mavros mavparam set NAV_RCL_ACT 0     # RC loss failsafe modea -> 0 (Not documented)
