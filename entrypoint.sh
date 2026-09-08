#!/bin/bash
set -e

mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/text_encoders /ComfyUI/models/vae /ComfyUI/models/upscale_models

download() {
  local repo="$1"
  local file="$2"
  local dest="$3"
  if [ -f "$dest" ]; then
    echo "Already present: $dest"
    return
  fi
  echo "Downloading $repo :: $file -> $dest"
  HF_XET_HIGH_PERFORMANCE=1 hf download "$repo" "$file" --local-dir "$(dirname "$dest")" --token "$HF_TOKEN"
  # hf download preserves the repo's internal path; move into place if nested
  local downloaded="$(dirname "$dest")/$file"
  if [ "$downloaded" != "$dest" ] && [ -f "$downloaded" ]; then
    mv "$downloaded" "$dest"
  fi
}

download "black-forest-labs/FLUX.2-klein-9b-fp8" "flux-2-klein-9b-fp8.safetensors" \
  "/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"

download "Comfy-Org/flux2-klein-9B" "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
  "/ComfyUI/models/text_encoders/qwen_3_8b_fp8mixed.safetensors"

download "black-forest-labs/FLUX.2-small-decoder" "full_encoder_small_decoder.safetensors" \
  "/ComfyUI/models/vae/full_encoder_small_decoder.safetensors"

# 1x skin-detail model: adds pores/micro-texture without changing resolution.
download "timothy692/1x-ITF-SkinDiffDetail-Lite-v1" "1x-ITF-SkinDiffDetail-Lite-v1.pth" \
  "/ComfyUI/models/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth"

echo "Starting ComfyUI..."
cd /ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 &

echo "Waiting for ComfyUI to be ready..."
until curl -sf http://127.0.0.1:8188/ > /dev/null; do
  sleep 1
done
echo "ComfyUI is ready."

echo "Starting the handler..."
cd /
python3 -u handler.py
