set -x
TASKS=$1 # tasks.json

project_in_container=/root/workspace

container_name=$( echo -n "${TASKS}" | md5sum | cut -c1-8)

docker run --name isaac-sim-${container_name} --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/docker/isaac-sim/downloads:/root/Downloads:rw \
    -v `pwd`:${project_in_container}:rw \
    -v /data1:/data1:rw \
    -v /data2:/data2:rw \
    isaac-sim:memo \
    -c "cd ${project_in_container} && /isaac-sim/python.sh -m src.utils.lib3d.render_templates_isaacsim ${TASKS}"