FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /ComfyUI

WORKDIR /ComfyUI
# The base image already ships torch 2.8.0+cu128; ComfyUI's requirements list
# torch/torchvision/torchaudio, and letting pip reinstall them added a ~2 GB
# layer for no gain. Image pull is the dominant cold-start cost, so strip them.
RUN grep -vE '^(torch|torchvision|torchaudio)\s*$' requirements.txt > /tmp/reqs.txt \
    && pip install --break-system-packages --no-cache-dir -r /tmp/reqs.txt \
    && rm -rf /root/.cache/pip

RUN pip install --break-system-packages --ignore-installed --no-cache-dir runpod websocket-client "huggingface_hub[hf_transfer]"

# PuLID-Flux2: identity injection from a reference photo, without LoRA training
# or the ReferenceLatent edit-mode path (which forces square output and loses
# likeness on refine — see olivia_terraza.png / olivia_v2.png). Experimental,
# single-maintainer project; kept isolated in its own custom_nodes dir so it
# can be ripped out cleanly if it doesn't hold up.
RUN git clone --depth 1 https://github.com/iFayens/ComfyUI-PuLID-Flux2.git \
      /ComfyUI/custom_nodes/ComfyUI-PuLID-Flux2 \
    && pip install --break-system-packages --no-cache-dir \
        insightface onnxruntime-gpu open-clip-torch "ml_dtypes>=0.5.0" \
    && rm -rf /root/.cache/pip
# ml_dtypes==0.3.2 was pinned first and broke: it's built against NumPy 1.x
# ABI while the base image ships NumPy 2.1.2, so `import onnx` (pulled in by
# insightface) died with "numpy.core.umath failed to import". Bumping to
# >=0.5.0 alone didn't fix it (still failed identically) — numpy itself was
# left in a mixed state (2.1.2 metadata, stale compiled _multiarray_umath)
# by the layered installs above. Force a clean reinstall of just numpy last,
# with no cache, so its metadata and compiled extension actually match.
RUN pip install --break-system-packages --no-cache-dir --force-reinstall --no-deps numpy \
    && rm -rf /root/.cache/pip

WORKDIR /
COPY workflow.json /workflow.json
COPY workflow_edit.json /workflow_edit.json
COPY handler.py /handler.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV SERVER_ADDRESS=127.0.0.1

CMD ["/entrypoint.sh"]
