#!/usr/bin/env bash

echo "worker-comfyui: Build version ${BUILD_VERSION:-unknown}"

# SYMLINK MODEL DIRS TO NETWORK VOLUME IF PRESENT
if [ -d /runpod-volume ]; then
    echo "worker-comfyui: Network volume detected, symlinking model dirs to /runpod-volume/models"
    for dir in diffusion_models clip vae loras controlnet; do
        mkdir -p "/runpod-volume/models/$dir"
        ln -sfn "/runpod-volume/models/$dir" "/comfyui/models/$dir"
    done
else
    echo "worker-comfyui: No network volume detected, using local model storage"
fi

# Download missing models via hf download (hf_xet chunk-based parallel transfers).
# Runs in foreground so models are guaranteed ready before ComfyUI starts.
echo "worker-comfyui: Validating models..."
/usr/local/bin/check-models.sh

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# DEBUG emits one line per quantized op per sampling step — ~3,600 lines for a 4-step job,
# logged from inside the sampling loop — and floods the console buffer hard enough to push
# the boot output out of a captured log.
: "${COMFY_LOG_LEVEL:=INFO}"

# Extra ComfyUI launch flags, set per endpoint. Memory, attention and precision options
# live here rather than baked in so they can be A/B tested from the console instead of
# costing a rebuild plus fleet-wide FlashBoot snapshot invalidation each.
: "${COMFY_EXTRA_ARGS:=}"

# Unquoted on purpose: the value is a flag list and must word-split.
# shellcheck disable=SC2086
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &
    echo $! > /tmp/comfyui.pid

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &
    echo $! > /tmp/comfyui.pid

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
