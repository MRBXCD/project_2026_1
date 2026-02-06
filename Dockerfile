# 1. Base Image
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 2. Set Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# 3. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    vim \
    nano \
    htop \
    tmux \
    screen \
    wget \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /workspace

# 5. Install Python General DL Tools
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn \
    tqdm \
    wandb \
    tensorboard 

# 6. Install LLM Core Dependencies
RUN pip install --no-cache-dir \
    transformers>=4.37.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.41.0 \
    peft>=0.8.0 \
    vllm>=0.3.0 \
    flash-attn \
    trl

# 7. Clone Your Target Repository
#RUN git clone https://github.com/harshita-chopra/misq-hf/workspace/misq-hf

# 8. Start Configuration
EXPOSE 8888
CMD ["/bin/bash"]