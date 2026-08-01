#!/usr/bin/env bash
# Model validation and downloading script for ComfyUI Qwen-Image-Edit-2511
# Uses `hf download` (hf_xet) for fast parallel chunk downloads from HuggingFace.

set -euo pipefail

# Use network volume for models and temp when available
if [ -d /runpod-volume ]; then
    MODEL_BASE="/runpod-volume/models"
    TMPDIR="/runpod-volume/tmp"
else
    MODEL_BASE="/comfyui/models"
    TMPDIR="/tmp"
fi
mkdir -p "$MODEL_BASE" "$TMPDIR"
export TMPDIR

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [check-models] [$level] $message" >&2
}

validate_model() {
    local model_path="$1"
    local model_name="$2"
    local full_path="${MODEL_BASE}/${model_path}"

    if [[ -f "$full_path" ]]; then
        local file_size=$(stat -c%s "$full_path" 2>/dev/null)
        if [[ $file_size -gt 0 ]]; then
            log "INFO" "✓ Model verified: $model_name ($(numfmt --to=iec $file_size))"
            return 0
        else
            log "ERROR" "✗ Model file exists but is empty: $model_name"
            return 1
        fi
    else
        log "WARN" "✗ Model missing: $model_name"
        return 1
    fi
}

download_model() {
    local repo_id="$1"
    local file_path="$2"
    local relative_path="$3"
    local model_name="$4"
    local max_retries=3
    local retry_count=0

    log "INFO" "Downloading $model_name..."

    while [[ $retry_count -lt $max_retries ]]; do
        local tmp_dir
        tmp_dir=$(mktemp -d)

        if hf download "$repo_id" "$file_path" --local-dir "$tmp_dir"; then
            local downloaded_file="$tmp_dir/$file_path"
            local target_dir="${MODEL_BASE}/$relative_path"
            mkdir -p "$target_dir"
            mv "$downloaded_file" "$target_dir/"
            rm -rf "$tmp_dir"
            log "INFO" "✓ Successfully downloaded $model_name"
            return 0
        else
            rm -rf "$tmp_dir"
            retry_count=$((retry_count + 1))
            if [[ $retry_count -lt $max_retries ]]; then
                log "WARN" "Download failed for $model_name (attempt $retry_count/$max_retries). Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done

    log "ERROR" "✗ Failed to download $model_name after $max_retries attempts"
    return 1
}

# Model configuration: repo_id|file_path_in_repo|target_subdir|display_name
declare -A MODELS=(
    ["diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"]="Comfy-Org/Qwen-Image-Edit_ComfyUI|split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors|diffusion_models|Qwen Edit Diffusion Model (fp8mixed)"
    ["clip/qwen_2.5_vl_7b_fp8_scaled.safetensors"]="Comfy-Org/Qwen-Image_ComfyUI|split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors|clip|Qwen CLIP Model"
    ["vae/qwen_image_vae.safetensors"]="Comfy-Org/Qwen-Image_ComfyUI|split_files/vae/qwen_image_vae.safetensors|vae|Qwen VAE Model"
    ["loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"]="lightx2v/Qwen-Image-Edit-2511-Lightning|Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors|loras|Qwen Edit Lightning 4-step LoRA"
)

main() {
    local missing_models=()
    local failed_downloads=()

    log "INFO" "Starting model validation..."

    for model_path in "${!MODELS[@]}"; do
        local model_info="${MODELS[$model_path]}"
        local model_name
        model_name=$(echo "$model_info" | cut -d'|' -f4)

        if ! validate_model "$model_path" "$model_name"; then
            missing_models+=("$model_path")
        fi
    done

    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log "INFO" "Found ${#missing_models[@]} missing models. Downloading via hf download (hf_xet)..."

        for model_path in "${missing_models[@]}"; do
            local model_info="${MODELS[$model_path]}"
            local repo_id
            local file_path
            local relative_path
            local model_name
            repo_id=$(echo "$model_info" | cut -d'|' -f1)
            file_path=$(echo "$model_info" | cut -d'|' -f2)
            relative_path=$(echo "$model_info" | cut -d'|' -f3)
            model_name=$(echo "$model_info" | cut -d'|' -f4)

            if ! download_model "$repo_id" "$file_path" "$relative_path" "$model_name"; then
                failed_downloads+=("$model_name")
            fi
        done

        if [[ ${#failed_downloads[@]} -gt 0 ]]; then
            log "WARN" "Failed to download ${#failed_downloads[@]} models: ${failed_downloads[*]}"
            log "WARN" "ComfyUI will continue but some features may not work properly"
        fi

        log "INFO" "Verifying downloaded models..."
        for model_path in "${missing_models[@]}"; do
            local model_info="${MODELS[$model_path]}"
            local model_name
            model_name=$(echo "$model_info" | cut -d'|' -f4)

            if [[ ! " ${failed_downloads[*]} " =~ " ${model_name} " ]]; then
                if ! validate_model "$model_path" "$model_name"; then
                    log "ERROR" "Download verification failed for $model_name"
                    failed_downloads+=("$model_name")
                fi
            fi
        done
    else
        log "INFO" "All required models are present and validated"
    fi

    log "INFO" "Model validation completed"
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
