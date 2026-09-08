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

WORKDIR /
COPY workflow.json /workflow.json
COPY workflow_edit.json /workflow_edit.json
COPY handler.py /handler.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV SERVER_ADDRESS=127.0.0.1

CMD ["/entrypoint.sh"]
