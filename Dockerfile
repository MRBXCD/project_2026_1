# 1. Base Image (NVIDIA PyTorch with CUDA support)
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

# 5. Install Python Dependencies (一次性安装全部依赖)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    # --- 核心 LLM 训练 ---
    transformers \
    accelerate \
    bitsandbytes \
    peft \
    trl \
    datasets \
    flash-attn \
    # --- 评估 ---
    "lighteval[accelerate]" \
    # --- 数据处理 ---
    pandas \
    numpy \
    # --- 合成数据 ---
    google-genai \
    # --- 实验管理 ---
    wandb \
    tensorboard \
    tqdm \
    # --- 工具 ---
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    scikit-learn \
    huggingface_hub

# 6. Start Configuration
EXPOSE 8888
CMD ["/bin/bash"]
