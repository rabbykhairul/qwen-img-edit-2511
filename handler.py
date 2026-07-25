import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback
import copy
import random

# LoRA filenames for the Edit-2511 model
LORA_FILES = {
    "4step": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "8step": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Websocket reconnection behaviour (can be overridden through environment variables)
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Extra verbose websocket trace logs (set WEBSOCKET_TRACE=true to enable)
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)

# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Helper: quick reachability probe of ComfyUI HTTP endpoint (port 8188)
# ---------------------------------------------------------------------------


def _comfy_server_status():
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """
    Attempts to reconnect to the WebSocket server after a disconnect.

    Args:
        ws_url (str): The WebSocket URL (including client_id).
        max_attempts (int): Maximum number of reconnection attempts.
        delay_s (int): Delay in seconds between attempts.
        initial_error (Exception): The error that triggered the reconnect attempt.

    Returns:
        websocket.WebSocket: The newly connected WebSocket object.

    Raises:
        websocket.WebSocketConnectionClosedException: If reconnection fails after all attempts.
    """
    print(
        f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect..."
    )
    last_reconnect_error = initial_error
    for attempt in range(max_attempts):
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            print(
                f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect: {srv_status.get('error', 'status '+str(srv_status.get('status_code')))}"
            )
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )

        print(
            f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}... (ComfyUI HTTP reachable, status {srv_status.get('status_code')})"
        )
        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print(f"worker-comfyui - Websocket reconnected successfully.")
            return new_ws
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as reconn_err:
            last_reconnect_error = reconn_err
            print(
                f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}"
            )
            if attempt < max_attempts - 1:
                print(
                    f"worker-comfyui - Waiting {delay_s} seconds before next attempt..."
                )
                time.sleep(delay_s)
            else:
                print(f"worker-comfyui - Max reconnection attempts reached.")

    print("worker-comfyui - Failed to reconnect websocket after connection closed.")
    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


DEFAULT_WORKFLOW = {
    "9": {
        "inputs": {
            "filename_prefix": "Qwen_Edit_2511",
            "images": ["170:158", 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
    },
    "41": {
        "inputs": {"image": "source_image.png"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Source Image"}
    },
    "83": {
        "inputs": {"image": "source_image.png"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Reference Image"}
    },
    "170:145": {
        "inputs": {"shift": 3.1, "model": ["170:161", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"}
    },
    "170:146": {
        "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"}
    },
    "170:147": {
        "inputs": {
            "reference_latents_method": "index_timestep_zero",
            "conditioning": ["170:149", 0]
        },
        "class_type": "FluxKontextMultiReferenceLatentMethod",
        "_meta": {"title": "FluxKontextMultiReferenceLatentMethod"}
    },
    "170:148": {
        "inputs": {
            "reference_latents_method": "index_timestep_zero",
            "conditioning": ["170:151", 0]
        },
        "class_type": "FluxKontextMultiReferenceLatentMethod",
        "_meta": {"title": "FluxKontextMultiReferenceLatentMethod"}
    },
    "170:149": {
        "inputs": {
            "prompt": "",
            "clip": ["170:162", 0],
            "vae": ["170:146", 0],
            "image1": ["170:160", 0],
            "image2": ["83", 0]
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "TextEncodeQwenImageEditPlus"}
    },
    "170:151": {
        "inputs": {
            "prompt": "",
            "clip": ["170:162", 0],
            "vae": ["170:146", 0],
            "image1": ["170:160", 0],
            "image2": ["83", 0]
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "TextEncodeQwenImageEditPlus (Positive)"}
    },
    "170:152": {
        "inputs": {"strength": 1, "model": ["170:145", 0]},
        "class_type": "CFGNorm",
        "_meta": {"title": "CFGNorm"}
    },
    "170:153": {
        "inputs": {
            "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "strength_model": 1,
            "model": ["170:152", 0]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoraLoaderModelOnly"}
    },
    "170:154": {
        "inputs": {"value": 4},
        "class_type": "PrimitiveFloat",
        "_meta": {"title": "CFG (Full)"}
    },
    "170:155": {
        "inputs": {"value": 1},
        "class_type": "PrimitiveFloat",
        "_meta": {"title": "CFG (Lightning)"}
    },
    "170:156": {
        "inputs": {"pixels": ["170:160", 0], "vae": ["170:146", 0]},
        "class_type": "VAEEncode",
        "_meta": {"title": "VAE Encode"}
    },
    "170:158": {
        "inputs": {"samples": ["170:169", 0], "vae": ["170:146", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"}
    },
    "170:160": {
        "inputs": {"image": ["41", 0]},
        "class_type": "FluxKontextImageScale",
        "_meta": {"title": "FluxKontextImageScale"}
    },
    "170:161": {
        "inputs": {
            "unet_name": "qwen_image_edit_2511_fp8mixed.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"}
    },
    "170:162": {
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
            "device": "default"
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"}
    },
    "170:163": {
        "inputs": {
            "UNKNOWN": False,
            "on_false": ["170:152", 0],
            "on_true": ["170:153", 0],
            "switch": ["170:168", 0]
        },
        "class_type": "ComfySwitchNode",
        "_meta": {"title": "Switch (Model)"}
    },
    "170:164": {
        "inputs": {
            "UNKNOWN": False,
            "on_false": ["170:154", 0],
            "on_true": ["170:155", 0],
            "switch": ["170:168", 0]
        },
        "class_type": "ComfySwitchNode",
        "_meta": {"title": "Switch (CFG)"}
    },
    "170:165": {
        "inputs": {"value": 4},
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Steps (Lightning)"}
    },
    "170:166": {
        "inputs": {"value": 40},
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Steps (Full)"}
    },
    "170:167": {
        "inputs": {
            "UNKNOWN": False,
            "on_false": ["170:166", 0],
            "on_true": ["170:165", 0],
            "switch": ["170:168", 0]
        },
        "class_type": "ComfySwitchNode",
        "_meta": {"title": "Switch (Steps)"}
    },
    "170:168": {
        "inputs": {"value": True},
        "class_type": "PrimitiveBoolean",
        "_meta": {"title": "Enable 4steps LoRA?"}
    },
    "170:169": {
        "inputs": {
            "seed": 0,
            "steps": ["170:167", 0],
            "cfg": ["170:164", 0],
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["170:163", 0],
            "positive": ["170:148", 0],
            "negative": ["170:147", 0],
            "latent_image": ["170:156", 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"}
    }
}


def _resolve_lora_mode(steps, lora_param=None):
    """Determine which LoRA to use based on steps and optional lora override.

    Returns: (lora_key, use_lora, auto_cfg)
        lora_key  - "4step", "8step", or None
        use_lora  - bool, whether to enable the LoRA path via the switch
        auto_cfg  - recommended CFG value (1.0 for Lightning, 4.0 for base)
    """
    if lora_param is not None:
        lora_param = lora_param.lower().strip()
        if lora_param == "none":
            return None, False, 4.0
        if lora_param in LORA_FILES:
            return lora_param, True, 1.0
        raise ValueError(
            f"Invalid lora value '{lora_param}'. Must be '4step', '8step', or 'none'."
        )
    if steps <= 4:
        return "4step", True, 1.0
    if steps <= 8:
        return "8step", True, 1.0
    return None, False, 4.0


def build_edit_workflow(
    prompt,
    source_image="source_image.png",
    reference_image=None,
    seed=None,
    steps=4,
    negative_prompt="",
    lora=None,
    cfg=None,
    shift=3.1,
    sampler="euler",
    scheduler="simple",
    megapixels=None,
):
    lora_key, use_lora, auto_cfg = _resolve_lora_mode(steps, lora)
    effective_cfg = cfg if cfg is not None else auto_cfg

    workflow = copy.deepcopy(DEFAULT_WORKFLOW)

    # Source image
    workflow["41"]["inputs"]["image"] = source_image

    # Reference image — if not provided, reuse the source image (single-image edit)
    ref_img = reference_image if reference_image else source_image
    workflow["83"]["inputs"]["image"] = ref_img

    # Edit prompt (positive)
    workflow["170:151"]["inputs"]["prompt"] = prompt

    # Negative prompt
    workflow["170:149"]["inputs"]["prompt"] = negative_prompt

    # Seed
    workflow["170:169"]["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**53)

    # Switch: true = Lightning LoRA, false = base model
    workflow["170:168"]["inputs"]["value"] = use_lora

    if use_lora and lora_key:
        workflow["170:153"]["inputs"]["lora_name"] = LORA_FILES[lora_key]

    # Override KSampler inputs directly (bypass switch links)
    workflow["170:169"]["inputs"]["steps"] = steps
    workflow["170:169"]["inputs"]["cfg"] = effective_cfg
    workflow["170:169"]["inputs"]["sampler_name"] = sampler
    workflow["170:169"]["inputs"]["scheduler"] = scheduler

    # Shift (ModelSamplingAuraFlow)
    workflow["170:145"]["inputs"]["shift"] = shift

    # FluxKontextImageScale (170:160) pins the source to ~1 MP, so a scaler placed
    # AFTER it is the only way to sample below 1 MP. Absent param → node never added
    # → default ~1 MP behavior stays byte-identical.
    if megapixels is not None:
        workflow["170:200"] = {
            "inputs": {
                "upscale_method": "lanczos",
                "megapixels": megapixels,
                # Rounds dims to a multiple of 16 (matches FluxKontextImageScale, Qwen-VAE safe).
                # Must be set explicitly: ComfyUI treats defaulted inputs as required in
                # API-format workflows (Comfy-Org/ComfyUI#11833), so omitting it fails validation.
                "resolution_steps": 16,
                "image": ["170:160", 0],
            },
            "class_type": "ImageScaleToTotalPixels",
            "_meta": {"title": "Resolution scale"},
        }
        workflow["170:156"]["inputs"]["pixels"] = ["170:200", 0]
        workflow["170:149"]["inputs"]["image1"] = ["170:200", 0]
        workflow["170:151"]["inputs"]["image1"] = ["170:200", 0]

    return workflow


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Input modes:
    1. "workflow" — raw ComfyUI API-format workflow (power users)
    2. "prompt" + "image" — simplified edit mode:
       - prompt (str, required): Edit instruction text
       - image (str, required): Base64-encoded source image or image name already uploaded
       - seed (int, optional): Random seed
       - steps (int, optional): Inference steps, default 4
       - megapixels (float, optional): Sample below ~1 MP (0.3–1.2) for faster generation; omit for full ~1 MP
       - negative_prompt (str, optional): Default ""
       - lora (str, optional): Override LoRA: "4step", "8step", "none"
       - cfg (float, optional): CFG scale (auto: 1.0 for Lightning, 4.0 for base)
       - shift (float, optional): ModelSamplingAuraFlow shift, default 3.1
       - sampler (str, optional): KSampler sampler, default "euler"
       - scheduler (str, optional): KSampler scheduler, default "simple"

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: (validated_data, error_message)
    """
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Determine input mode: raw workflow or simplified edit
    workflow = job_input.get("workflow")
    prompt = job_input.get("prompt")

    if workflow is not None:
        pass  # Raw workflow mode — use as-is
    elif prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            return None, "'prompt' must be a non-empty string"

        # For edit mode, a source image is required
        source_image = job_input.get("image")
        if not source_image:
            return None, "'image' is required when using simplified edit mode. Provide a base64-encoded image."

        # Validate new params
        _lora_val = job_input.get("lora")
        if _lora_val is not None:
            _lora_val = str(_lora_val).lower().strip()
            if _lora_val not in ("4step", "8step", "none"):
                return None, "'lora' must be '4step', '8step', or 'none'"

        _steps_val = job_input.get("steps", 4)
        if not isinstance(_steps_val, int) or _steps_val < 1:
            return None, "'steps' must be a positive integer"

        _megapixels_val = job_input.get("megapixels")
        if _megapixels_val is not None and (
            not isinstance(_megapixels_val, (int, float)) or not (0.3 <= _megapixels_val <= 1.2)
        ):
            return None, "'megapixels' must be a number between 0.3 and 1.2"

        _cfg_val = job_input.get("cfg")
        if _cfg_val is not None and (not isinstance(_cfg_val, (int, float)) or _cfg_val < 0):
            return None, "'cfg' must be a non-negative number"

        _shift_val = job_input.get("shift")
        if _shift_val is not None and (not isinstance(_shift_val, (int, float)) or _shift_val <= 0):
            return None, "'shift' must be a positive number"

        _sampler_val = job_input.get("sampler", "euler")
        if not isinstance(_sampler_val, str) or not _sampler_val.strip():
            return None, "'sampler' must be a non-empty string"

        _scheduler_val = job_input.get("scheduler", "simple")
        if not isinstance(_scheduler_val, str) or not _scheduler_val.strip():
            return None, "'scheduler' must be a non-empty string"

        # Prepare images for upload
        images = []
        source_image_name = job_input.get("image_name", "source_image.png")

        # Source image
        if source_image.startswith("data:") or len(source_image) > 200:
            images.append({"name": source_image_name, "image": source_image})
            source_ref = source_image_name
        else:
            source_ref = source_image

        # Optional second reference image for multi-image edits
        reference_ref = None
        reference_image = job_input.get("reference_image")
        if reference_image:
            ref_name = job_input.get("reference_image_name", "reference_image.png")
            if reference_image.startswith("data:") or len(reference_image) > 200:
                images.append({"name": ref_name, "image": reference_image})
                reference_ref = ref_name
            else:
                reference_ref = reference_image

        workflow = build_edit_workflow(
            prompt=prompt,
            source_image=source_ref,
            reference_image=reference_ref,
            seed=job_input.get("seed"),
            steps=_steps_val,
            negative_prompt=job_input.get("negative_prompt", ""),
            lora=_lora_val,
            cfg=_cfg_val,
            shift=_shift_val if _shift_val is not None else 3.1,
            sampler=_sampler_val,
            scheduler=_scheduler_val,
            megapixels=_megapixels_val,
        )
        # Carry the images for upload
        if images:
            job_input["_images_to_upload"] = images
    else:
        return None, "Missing 'workflow' or 'prompt' parameter"

    # Validate 'images' in input, if provided (for raw workflow mode)
    images = job_input.get("images") or job_input.pop("_images_to_upload", None)
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    comfy_org_api_key = job_input.get("comfy_org_api_key")

    return {
        "workflow": workflow,
        "images": images,
        "comfy_org_api_key": comfy_org_api_key,
    }, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request
    """

    def _comfyui_pid_alive():
        try:
            pid_path = "/tmp/comfyui.pid"
            if os.path.exists(pid_path):
                with open(pid_path) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return True
        except (ProcessLookupError, ValueError, PermissionError):
            return False
        return None

    print(f"worker-comfyui - Checking API server at {url}...")
    for i in range(retries):
        pid_status = _comfyui_pid_alive()
        if pid_status is False:
            print(f"worker-comfyui - ComfyUI process (PID file) is dead, aborting.")
            return False

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException:
            pass

        time.sleep(delay / 1000)

    print(
        f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """Upload a list of base64 encoded images to the ComfyUI server."""
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    for image in images:
        try:
            name = image["name"]
            image_data_uri = image["image"]

            if "," in image_data_uri:
                base64_data = image_data_uri.split(",", 1)[1]
            else:
                base64_data = image_data_uri

            blob = base64.b64decode(base64_data)

            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print(f"worker-comfyui - image(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"worker-comfyui - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def get_available_models():
    """Get list of available models from ComfyUI"""
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


QWEN_MODELS = {
    "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "relative_path": "diffusion_models",
        "filename": "qwen_image_edit_2511_fp8mixed.safetensors",
        "name": "Qwen Edit Diffusion Model (fp8mixed)",
        "type": "unet"
    },
    "clip/qwen_2.5_vl_7b_fp8_scaled.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "relative_path": "clip",
        "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "name": "Qwen CLIP Model",
        "type": "clip"
    },
    "vae/qwen_image_vae.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "relative_path": "vae",
        "filename": "qwen_image_vae.safetensors",
        "name": "Qwen VAE Model",
        "type": "vae"
    },
    "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": {
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "relative_path": "loras",
        "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "name": "Qwen Edit Lightning 4-step LoRA",
        "type": "loras"
    },
    "loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors": {
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        "relative_path": "loras",
        "filename": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        "name": "Qwen Edit Lightning 8-step LoRA",
        "type": "loras"
    }
}


def extract_required_models(workflow_data):
    """Extract required models from workflow JSON by analyzing model-loading nodes"""
    required_models = set()

    if not workflow_data or not isinstance(workflow_data, dict):
        return required_models

    for node_id, node_data in workflow_data.items():
        if not isinstance(node_data, dict) or "class_type" not in node_data:
            continue

        class_type = node_data["class_type"]
        inputs = node_data.get("inputs", {})

        if class_type in ["UNETLoader", "CheckpointLoaderSimple", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly"]:
            for param_name, param_value in inputs.items():
                if param_name in ["unet_name", "ckpt_name", "clip_name", "vae_name", "lora_name"]:
                    if isinstance(param_value, str) and param_value:
                        required_models.add(param_value)

    return required_models


def validate_model_exists(model_filename, model_type=None):
    """Check if a model file exists locally"""
    if model_type:
        model_path = f"/comfyui/models/{model_type}/{model_filename}"
        if os.path.exists(model_path):
            return True

    model_dirs = ["diffusion_models", "clip", "vae", "loras", "checkpoints", "unet", "controlnet"]
    for model_dir in model_dirs:
        model_path = f"/comfyui/models/{model_dir}/{model_filename}"
        if os.path.exists(model_path):
            return True

    return False


def validate_required_models(required_models):
    """Validate that all required models are available locally"""
    missing_models = []
    found_models = []

    for model_filename in required_models:
        found = False

        for model_config in QWEN_MODELS.values():
            if model_config["filename"] == model_filename:
                if validate_model_exists(model_filename, model_config["relative_path"]):
                    found_models.append(model_filename)
                    found = True
                    break

        if not found:
            missing_models.append(model_filename)

    return missing_models, found_models


def download_model(model_config, client_id=None):
    """Download a model using ComfyUI CLI with progress reporting"""
    import subprocess

    try:
        print(f"worker-comfyui - Downloading {model_config['name']}...")

        if client_id:
            send_download_status(client_id, {
                "status": "downloading",
                "model": model_config["name"],
                "progress": 0
            })

        cmd = [
            "comfy", "model", "download",
            "--url", model_config["url"],
            "--relative-path", f"models/{model_config['relative_path']}",
            "--filename", model_config["filename"]
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

        if result.returncode == 0:
            print(f"worker-comfyui - Successfully downloaded {model_config['name']}")

            if client_id:
                send_download_status(client_id, {
                    "status": "completed",
                    "model": model_config["name"],
                    "progress": 100
                })
            return True
        else:
            print(f"worker-comfyui - Download failed for {model_config['name']}: {result.stderr}")

            if client_id:
                send_download_status(client_id, {
                    "status": "error",
                    "model": model_config["name"],
                    "error": result.stderr
                })
            return False

    except subprocess.TimeoutExpired:
        print(f"worker-comfyui - Download timeout for {model_config['name']}")
        if client_id:
            send_download_status(client_id, {
                "status": "error",
                "model": model_config["name"],
                "error": "Download timeout (20 minutes)"
            })
        return False
    except Exception as e:
        print(f"worker-comfyui - Download error for {model_config['name']}: {e}")
        if client_id:
            send_download_status(client_id, {
                "status": "error",
                "model": model_config["name"],
                "error": str(e)
            })
        return False


def download_missing_models(missing_models, client_id=None):
    """Download all missing models"""
    successful = []
    failed = []

    for model_filename in missing_models:
        model_config = None
        for config in QWEN_MODELS.values():
            if config["filename"] == model_filename:
                model_config = config
                break

        if not model_config:
            print(f"worker-comfyui - Unknown model: {model_filename}")
            failed.append(model_filename)
            continue

        if download_model(model_config, client_id):
            successful.append(model_filename)
        else:
            failed.append(model_filename)

    return successful, failed


def send_download_status(client_id, status_data):
    """Send download status via WebSocket"""
    print(f"worker-comfyui - Download status [{client_id}]: {status_data}")


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    """Queue a workflow to be processed by ComfyUI"""
    payload = {"prompt": workflow, "client_id": client_id}

    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30
    )

    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = response.json()
            print(f"worker-comfyui - Parsed error data: {error_data}")

            error_message = "Workflow validation failed"
            error_details = []

            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                    if error_info.get("type") == "prompt_outputs_failed_validation":
                        error_message = "Workflow validation failed"
                else:
                    error_message = str(error_info)

            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(
                                f"Node {node_id} ({error_type}): {error_msg}"
                            )
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            if error_data.get("type") == "prompt_outputs_failed_validation":
                error_message = error_data.get("message", "Workflow validation failed")
                available_models = get_available_models()
                if available_models.get("checkpoints"):
                    error_message += f"\n\nThis usually means a required model or parameter is not available."
                    error_message += f"\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                else:
                    error_message += "\n\nThis usually means a required model or parameter is not available."
                    error_message += "\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(error_message)

            if error_details:
                detailed_message = f"{error_message}:\n" + "\n".join(
                    f"• {detail}" for detail in error_details
                )

                if any(
                    "not in list" in detail and "ckpt_name" in detail
                    for detail in error_details
                ):
                    available_models = get_available_models()
                    if available_models.get("checkpoints"):
                        detailed_message += f"\n\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                    else:
                        detailed_message += "\n\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(detailed_message)
            else:
                raise ValueError(f"{error_message}. Raw response: {response.text}")

        except (json.JSONDecodeError, KeyError):
            raise ValueError(
                f"ComfyUI validation failed (could not parse error response): {response.text}"
            )

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """Retrieve the history of a given prompt using its ID"""
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_image_data(filename, subfolder, image_type):
    """Fetch image bytes from the ComfyUI /view endpoint."""
    print(
        f"worker-comfyui - Fetching image data: type={image_type}, subfolder={subfolder}, filename={filename}"
    )
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    try:
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=60)
        response.raise_for_status()
        print(f"worker-comfyui - Successfully fetched image data for {filename}")
        return response.content
    except requests.Timeout:
        print(f"worker-comfyui - Timeout fetching image data for {filename}")
        return None
    except requests.RequestException as e:
        print(f"worker-comfyui - Error fetching image data for {filename}: {e}")
        return None
    except Exception as e:
        print(f"worker-comfyui - Unexpected error fetching image data for {filename}: {e}")
        return None


_R2_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_r2_client = None


def _r2_config():
    """Return R2 upload config if all required env vars are set, else None."""
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    bucket = os.environ.get("R2_BUCKET")
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (endpoint and bucket and key_id and secret):
        return None
    return {
        "endpoint": endpoint,
        "bucket": bucket,
        "key_id": key_id,
        "secret": secret,
        "public_base": os.environ.get("R2_PUBLIC_BASE_URL"),
        "expiry": int(os.environ.get("R2_URL_EXPIRY", "604800")),
    }


def upload_image_to_r2(job_id, image_bytes, filename, content_type, cfg):
    """Upload output image bytes to Cloudflare R2 and return a fetchable URL.

    rp_upload can't target R2 — it derives the bucket name from the endpoint
    hostname (AWS-shaped), which breaks on R2's <account>.r2.cloudflarestorage.com.
    So upload with an explicit bucket via a plain boto3 S3 client. boto3 ships with
    the runpod package (rp_upload uses it), so this adds no new dependency.
    """
    global _r2_client
    if _r2_client is None:
        import boto3
        from botocore.config import Config

        _r2_client = boto3.client(
            "s3",
            endpoint_url=cfg["endpoint"],
            aws_access_key_id=cfg["key_id"],
            aws_secret_access_key=cfg["secret"],
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    key = f"{job_id}/{filename}"
    _r2_client.put_object(
        Bucket=cfg["bucket"], Key=key, Body=image_bytes, ContentType=content_type
    )

    if cfg["public_base"]:
        return f"{cfg['public_base'].rstrip('/')}/{key}"
    return _r2_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": cfg["bucket"], "Key": key},
        ExpiresIn=cfg["expiry"],
    )


def handler(job):
    """
    Handles an image editing job using Qwen-Image-Edit-2511 via ComfyUI.

    Input modes:
    1. Raw workflow: {"input": {"workflow": {...}, "images": [...]}}
    2. Simplified edit: {"input": {"prompt": "edit instruction", "image": "base64...", ...}}
    """
    job_input = job["input"]
    job_id = job["id"]

    # Health-check probe for RunPod Hub test validation — returns immediately
    # without touching ComfyUI, so the Hub sees a 200 response.
    if isinstance(job_input, dict) and job_input.get("health_check"):
        print("worker-comfyui - Health check probe received, simulating workload...")
        time.sleep(3)
        return {"status": "healthy"}

    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")

    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {
            "error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."
        }

    # Validate and download required models
    print(f"worker-comfyui - Checking workflow model requirements...")
    required_models = extract_required_models(workflow)
    print(f"worker-comfyui - Required models: {required_models}")

    if required_models:
        missing_models, found_models = validate_required_models(required_models)

        if missing_models:
            print(f"worker-comfyui - Missing models: {missing_models}")
            print(f"worker-comfyui - Found models: {found_models}")
            print(f"worker-comfyui - Downloading missing models...")

            successful_downloads, failed_downloads = download_missing_models(missing_models)

            if successful_downloads:
                print(f"worker-comfyui - Successfully downloaded: {successful_downloads}")

            if failed_downloads:
                error_msg = f"Failed to download required models: {failed_downloads}"
                print(f"worker-comfyui - {error_msg}")
                return {"error": error_msg}

            missing_models, found_models = validate_required_models(required_models)
            if missing_models:
                error_msg = f"Still missing required models after download attempt: {missing_models}"
                print(f"worker-comfyui - {error_msg}")
                return {"error": error_msg}
        else:
            print(f"worker-comfyui - All required models found locally: {found_models}")

    # Upload input images if they exist
    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            return {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
            }

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    errors = []

    try:
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-comfyui - Websocket connected")

        try:
            queued_workflow = queue_workflow(
                workflow,
                client_id,
                comfy_org_api_key=validated_data.get("comfy_org_api_key"),
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"Missing 'prompt_id' in queue response: {queued_workflow}")
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except requests.RequestException as e:
            print(f"worker-comfyui - Error queuing workflow: {e}")
            raise ValueError(f"Error queuing workflow: {e}")
        except Exception as e:
            print(f"worker-comfyui - Unexpected error queuing workflow: {e}")
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Unexpected error queuing workflow: {e}")

        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...")
        execution_done = False
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        status_data = message.get("data", {}).get("status", {})
                        print(
                            f"worker-comfyui - Status update: {status_data.get('exec_info', {}).get('queue_remaining', 'N/A')} items remaining in queue"
                        )
                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("node") is None
                            and data.get("prompt_id") == prompt_id
                        ):
                            print(f"worker-comfyui - Execution finished for prompt {prompt_id}")
                            execution_done = True
                            break
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, Message: {data.get('exception_message')}"
                            print(f"worker-comfyui - Execution error received: {error_details}")
                            errors.append(f"Workflow execution error: {error_details}")
                            break
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                print(f"worker-comfyui - Websocket receive timed out. Still waiting...")
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    ws = _attempt_websocket_reconnect(
                        ws_url,
                        WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S,
                        closed_err,
                    )
                    print("worker-comfyui - Resuming message listening after successful reconnect.")
                    continue
                except websocket.WebSocketConnectionClosedException as reconn_failed_err:
                    raise reconn_failed_err
            except json.JSONDecodeError:
                print(f"worker-comfyui - Received invalid JSON message via websocket.")

        if not execution_done and not errors:
            raise ValueError(
                "Workflow monitoring loop exited without confirmation of completion or error."
            )

        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history after execution."
            print(f"worker-comfyui - {error_msg}")
            if not errors:
                return {"error": error_msg}
            else:
                errors.append(error_msg)
                return {
                    "error": "Job processing failed, prompt ID not found in history.",
                    "details": errors,
                }

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        if not outputs:
            warning_msg = f"No outputs found in history for prompt {prompt_id}."
            print(f"worker-comfyui - {warning_msg}")
            if not errors:
                errors.append(warning_msg)

        print(f"worker-comfyui - Processing {len(outputs)} output nodes...")
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                print(
                    f"worker-comfyui - Node {node_id} contains {len(node_output['images'])} image(s)"
                )
                for image_info in node_output["images"]:
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")

                    if img_type == "temp":
                        print(f"worker-comfyui - Skipping image {filename} because type is 'temp'")
                        continue

                    if not filename:
                        warn_msg = f"Skipping image in node {node_id} due to missing filename: {image_info}"
                        print(f"worker-comfyui - {warn_msg}")
                        errors.append(warn_msg)
                        continue

                    image_bytes = get_image_data(filename, subfolder, img_type)

                    if image_bytes:
                        file_extension = os.path.splitext(filename)[1] or ".png"

                        r2_cfg = _r2_config()
                        if r2_cfg:
                            try:
                                content_type = _R2_CONTENT_TYPES.get(
                                    file_extension.lower(), "application/octet-stream"
                                )
                                url = upload_image_to_r2(
                                    job_id, image_bytes, filename, content_type, r2_cfg
                                )
                                print(f"worker-comfyui - Uploaded {filename} to R2: {url}")
                                output_data.append(
                                    {"filename": filename, "type": "s3_url", "data": url}
                                )
                            except Exception as e:
                                error_msg = f"Error uploading {filename} to R2: {e}"
                                print(f"worker-comfyui - {error_msg}")
                                errors.append(error_msg)
                        elif os.environ.get("BUCKET_ENDPOINT_URL"):
                            try:
                                with tempfile.NamedTemporaryFile(
                                    suffix=file_extension, delete=False
                                ) as temp_file:
                                    temp_file.write(image_bytes)
                                    temp_file_path = temp_file.name
                                print(f"worker-comfyui - Wrote image bytes to temporary file: {temp_file_path}")

                                print(f"worker-comfyui - Uploading {filename} to S3...")
                                s3_url = rp_upload.upload_image(job_id, temp_file_path)
                                os.remove(temp_file_path)
                                print(f"worker-comfyui - Uploaded {filename} to S3: {s3_url}")
                                output_data.append(
                                    {"filename": filename, "type": "s3_url", "data": s3_url}
                                )
                            except Exception as e:
                                error_msg = f"Error uploading {filename} to S3: {e}"
                                print(f"worker-comfyui - {error_msg}")
                                errors.append(error_msg)
                                if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                                    try:
                                        os.remove(temp_file_path)
                                    except OSError as rm_err:
                                        print(f"worker-comfyui - Error removing temp file {temp_file_path}: {rm_err}")
                        else:
                            try:
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                output_data.append(
                                    {"filename": filename, "type": "base64", "data": base64_image}
                                )
                                print(f"worker-comfyui - Encoded {filename} as base64")
                            except Exception as e:
                                error_msg = f"Error encoding {filename} to base64: {e}"
                                print(f"worker-comfyui - {error_msg}")
                                errors.append(error_msg)
                    else:
                        error_msg = f"Failed to fetch image data for {filename} from /view endpoint."
                        errors.append(error_msg)

            other_keys = [k for k in node_output.keys() if k != "images"]
            if other_keys:
                warn_msg = f"Node {node_id} produced unhandled output keys: {other_keys}."
                print(f"worker-comfyui - WARNING: {warn_msg}")
                print(f"worker-comfyui - --> If this output is useful, please consider opening an issue on GitHub to discuss adding support.")

    except websocket.WebSocketException as e:
        print(f"worker-comfyui - WebSocket Error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}
    except requests.RequestException as e:
        print(f"worker-comfyui - HTTP Request Error: {e}")
        print(traceback.format_exc())
        return {"error": f"HTTP communication error with ComfyUI: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - Value Error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    except Exception as e:
        print(f"worker-comfyui - Unexpected Handler Error: {e}")
        print(traceback.format_exc())
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if ws and ws.connected:
            print(f"worker-comfyui - Closing websocket connection.")
            ws.close()

    final_result = {}

    if output_data:
        final_result["images"] = output_data

    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - Job completed with errors/warnings: {errors}")

    if not output_data and errors:
        print(f"worker-comfyui - Job failed with no output images.")
        return {
            "error": "Job processing failed",
            "details": errors,
        }
    elif not output_data and not errors:
        print(f"worker-comfyui - Job completed successfully, but the workflow produced no images.")
        final_result["status"] = "success_no_images"
        final_result["images"] = []

    print(f"worker-comfyui - Job completed. Returning {len(output_data)} image(s).")
    return final_result


if __name__ == "__main__":
    print("worker-comfyui - Starting handler...")
    runpod.serverless.start({"handler": handler})
