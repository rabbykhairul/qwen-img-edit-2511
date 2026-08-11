# CLAUDE.md

Qwen Image Edit 2511 — image editing RunPod Serverless endpoint. Accepts a source image + text instruction, with optional reference image for multi-image edits. Lightning LoRA (4-step / 8-step) support. See root `../CLAUDE.md` for monorepo-wide context.

## Handler Parameters (Simplified Input)

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | str | required | Edit instruction |
| `image` | str | required | Base64-encoded source image |
| `reference_image` | str | source | Separate reference image; if omitted, source is reused |
| `reference_image_name` | str | `"reference_image.png"` | Filename for the reference image |
| `seed` | int | random | |
| `steps` | int | 4 | Also determines LoRA mode |
| `negative_prompt` | str | `""` | |
| `lora` | str | auto | `"4step"`, `"8step"`, or `"none"` |
| `cfg` | float | auto | Auto: 1.0 for Lightning, 4.0 for base |
| `shift` | float | 3.1 | |
| `sampler` | str | `"euler"` | |
| `scheduler` | str | `"simple"` | |

No `width`/`height`/`batch_size` — dimensions come from the source image.

## Key Difference from 2512

Uses `FluxKontextMultiReferenceLatentMethod` nodes for conditioning instead of standard `CLIPTextEncode`. Two `LoadImage` nodes: `"41"` (source) and `"83"` (reference). If no reference image is provided, the source image is reused for both.

## Models

| File | Relative path |
|------|---------------|
| Diffusion (fp8mixed) | `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` |
| Text encoder | `clip/qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAE | `vae/qwen_image_vae.safetensors` |
| 4-step LoRA | `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |
| 8-step LoRA | `loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` |

Base path: `/runpod-volume/models/` when network volume attached, `/comfyui/models/` otherwise. `start.sh` symlinks `/comfyui/models/X` → volume path. `check-models.sh` downloads to `$MODEL_BASE`.

Note: LoRA files use bf16 precision (not fp32 like 2512). **Sync requirement:** `QWEN_MODELS` dict in `handler.py` and `MODELS` array in `src/check-models.sh` must match.

## Key Node IDs

Not portable to other projects:
- KSampler: `"170:169"` — steps, cfg, sampler_name, scheduler, seed
- Source LoadImage: `"41"` — image input
- Reference LoadImage: `"83"` — reference image (or same as source)
- LoRA switch: `"170:168"` — boolean enables Lightning path
- LoraLoaderModelOnly: `"170:153"` — lora_name
- Positive edit text: `"170:151"`, Negative text: `"170:149"` — swapped for a `ConditioningZeroOut` off the positive whenever CFG resolves to 1.0, since the sampler skips the uncond pass entirely there
- Shift: `"170:145"` — ModelSamplingAuraFlow
- FluxKontextImageScale: `"76"` — scales the source image

## Build & Deploy

```bash
docker build -t nyxcoolminds/qwen-img:qwen-edit-2511 .
docker push nyxcoolminds/qwen-img:qwen-edit-2511
docker rmi nyxcoolminds/qwen-img:qwen-edit-2511 && docker builder prune -af
```

Release: `gh release create v1.X.Y --title "v1.X.Y" --notes "description"`

## RunPod Hub Tests

`.runpod/tests.json` — health_check only (300s timeout, RTX 4090, CUDA 12.8).
