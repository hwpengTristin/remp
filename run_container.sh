xhost +
docker run -it --rm -e DISPLAY=$DISPLAY -v $(pwd):/root/remp -v /tmp/.X11-unix:/tmp/.X11-unix arcldocker/remp:latest bash
xhost -
