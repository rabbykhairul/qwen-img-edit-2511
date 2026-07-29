# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with all dependencies
FROM ${BASE_IMAGE} AS base

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG COMFY_CUSTOM_NODES=comfyui-image-saver
ARG BUILD_VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python 3.12 (native in Ubuntu 24.04), git, and runtime libs
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv and create venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia --fast-deps; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia --fast-deps; \
    fi

# Install ComfyUI runtime requirements
RUN uv pip install -r /comfyui/requirements.txt

# cu130 is what unlocks comfy-kitchen's `cuda` and `triton` backends. On cu128 both report
# available:True disabled:True, and every quantized op falls back to eager dispatch —
# ~3,600 per-layer dequantize calls for a 4-step job. The cost is a CUDA 13.0 driver floor
# (580+), a major-version bump with no minor-version compatibility to fall back on.
ARG TORCH_CUDA_CHANNEL=cu130
RUN uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}

# Support for the network volume
ADD src/extra_model_paths.yaml /comfyui/

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client huggingface_hub piexif

# Add custom node install script
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during custom node installs
ENV PIP_NO_INPUT=1

# Install custom nodes
RUN if [ -n "${COMFY_CUSTOM_NODES}" ]; then \
      echo "Installing custom ComfyUI nodes: ${COMFY_CUSTOM_NODES}" && \
      /usr/local/bin/comfy-node-install ${COMFY_CUSTOM_NODES} || \
      (echo "Failed to install custom nodes" && exit 1); \
    else \
      echo "No custom ComfyUI nodes to install"; \
    fi

WORKDIR /comfyui

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

COPY src/check-models.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/check-models.sh

# Go back to root for handler files
WORKDIR /

ADD src/start.sh test_input.json ./
RUN chmod +x /start.sh

# Without this every cold container regenerates __pycache__ for torch and ComfyUI into the
# writable layer on first import, and throws it away when the worker dies.
RUN python -m compileall -q /opt/venv/lib/python3.12/site-packages /comfyui || true

# ComfyUI creates its asset DB on first start and walks six Alembic revisions to get there;
# baking the finished DB skips that on every cold boot. Best-effort — a ComfyUI version that
# doesn't support this exit-after-init path just leaves the DB to be built at runtime.
RUN timeout 300 python /comfyui/main.py --cpu --disable-auto-launch --quick-test-for-ci >/dev/null 2>&1 || true; \
    test -f /comfyui/user/comfyui.db \
      && echo "comfyui.db seeded at build" \
      || echo "comfyui.db NOT seeded — migrations will run at boot"

# nvidia/cuda ships forward-compat driver stubs here and the runtime prefers them over the
# host driver whenever the host is older. NVIDIA supports that path only on data-center
# GPUs, so on GeForce it fails with CUDA error 804 after the container has already started.
RUN rm -rf /usr/local/cuda/compat

# Must track TORCH_CUDA_CHANNEL — cu130 needs a 13.0 driver, cu128 needs 12.0.
ARG REQUIRE_CUDA=13.0
ENV NVIDIA_REQUIRE_CUDA="cuda>=${REQUIRE_CUDA}"

# Enable high-performance downloads from HuggingFace (hf_xet chunk-based parallel transfers).
ENV HF_XET_HIGH_PERFORMANCE=1

# Stamp build version for runtime identification
ENV BUILD_VERSION=${BUILD_VERSION}

CMD ["/start.sh"]

# Stage 2: Final image (models baked in at build time)
FROM base AS final

# Bake all models into the image so the worker needs no runtime download and no
# network volume. /runpod-volume is absent during build, so check-models.sh writes
# to /comfyui/models. The script exits 0 even on a failed download, so assert every
# file is present to fail the build on an incomplete bake.
ARG HF_TOKEN=""
RUN HF_TOKEN="${HF_TOKEN}" /usr/local/bin/check-models.sh \
    && test -f /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
    && test -f /comfyui/models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    && test -f /comfyui/models/vae/qwen_image_vae.safetensors \
    && test -f /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    && test -f /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors

# handler.py is copied LAST — after the model bake — so a handler-only change reuses the
# cached ~30 GB model layers instead of re-baking them, and RunPod only re-pulls this
# tiny layer instead of the whole image.
COPY handler.py /handler.py
