docker run --gpus all \ 
--env NVIDIA_DISABLE_REQUIRE=1 \
-it \
--network=host \
--name sam2 \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v $DIR:$DIR \
-v /home:/home \
-v /mnt:/mnt \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /tmp:/tmp \
--ipc=host \
-e DISPLAY=${DISPLAY} \
-e GIT_INDEX_FILE trlc/sam2 bash \
-c "cd $DIR && bash" \

# docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY --gpus all sam2:latest bash

