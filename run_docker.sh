# For graphics
xhost +

docker run \
	--interactive \
	--tty \
	--rm \
	--env DISPLAY=$DISPLAY \
	--privileged \
	--volume /tmp/.X11-unix:/tmp/.X11-unix \
	--volume $(pwd)/../g1_description:/home/forest_ws/src/g1_description \
	--volume $(pwd)/code:/home/forest_ws/src/g1_opensot \
	opensot
