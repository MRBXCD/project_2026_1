cd /home/mrb/projects/proj_2026_1

docker run -it -d \
  --gpus all \
  --name my_llm_container \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  llm-dev-env:v1