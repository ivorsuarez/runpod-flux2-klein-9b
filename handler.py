import runpod
import os
import json
import uuid
import base64
import logging
import urllib.request
import urllib.parse
import time
import websocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_T2I = os.path.join(HERE, "workflow.json")
WORKFLOW_EDIT = os.path.join(HERE, "workflow_edit.json")

# --- text-to-image graph ---------------------------------------------------
T2I = {
    "positive": "75:74", "negative": "75:67", "seed": "75:73",
    "width": "75:68", "height": "75:69", "steps": "75:62",
    "cfg": "75:63", "nag": "100", "base_image": "75:65",
    "refine_resize": "101", "refine_sched": "103", "refine_sampler": "104",
    "refine_seed": "105", "refine_image": "107", "save": "9",
    "refine_nodes": ["101", "102", "103", "104", "105", "106", "107"],
    "skin_nodes": ["108", "109"],
}

# --- reference-edit graph --------------------------------------------------
EDIT = {
    "positive": "20", "seed": "44", "steps": "42", "cfg": "45",
    "ref_load": "10", "ref_scale": "11", "ref_encode": "12",
    "pos_ref": "30", "neg_ref": "31", "guider": "45",
    "base_image": "47",
    "refine_resize": "50", "refine_sched": "52", "refine_sampler": "53",
    "refine_seed": "54", "refine_image": "56", "save": "9",
    "refine_nodes": ["50", "51", "52", "53", "54", "55", "56"],
    "skin_nodes": ["60", "61"],
}

DEFAULT_NEGATIVE = (
    "plastic skin, airbrushed, waxy, smooth skin, cgi, 3d render, illustration, "
    "painting, oversaturated, overexposed, blurry, low detail"
)
DETAIL_SUFFIX = "8K, intricate details"


def load_workflow(path):
    with open(path, "r") as f:
        return json.load(f)


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    data = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(url, data=data)).read())
    except urllib.error.HTTPError as exc:
        # ComfyUI puts the actual validation failure (which node, which input)
        # in the response body; without it a 400 says nothing useful.
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:
            detail = "<no body>"
        logger.error("ComfyUI rejected the prompt (%s): %s", exc.code, detail)
        raise RuntimeError(f"ComfyUI rejected the prompt ({exc.code}): {detail}") from None


def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    with urllib.request.urlopen(f"{url}?{params}") as response:
        return response.read()


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def fetch_bytes(source):
    """A reference may arrive as an http(s) URL, a data: URI or raw base64."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=60) as r:
            return r.read()
    if source.startswith("data:"):
        source = source.split(",", 1)[1]
    return base64.b64decode(source)


def upload_image(data, filename):
    """POST to ComfyUI's /upload/image; returns the name LoadImage expects."""
    boundary = f"----comfy{uuid.uuid4().hex}"
    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n')
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"http://{server_address}:8188/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    name = resp["name"]
    if resp.get("subfolder"):
        name = f"{resp['subfolder']}/{name}"
    logger.info("Uploaded reference -> %s (%d bytes)", name, len(data))
    return name


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break

    history = get_history(prompt_id)[prompt_id]
    images_output = []
    for node_output in history["outputs"].values():
        for image in node_output.get("images", []):
            raw = get_image(image["filename"], image["subfolder"], image["type"])
            images_output.append(base64.b64encode(raw).decode("utf-8"))
    return images_output


def wait_for_server(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{server_address}:8188/", timeout=5)
            return
        except Exception:
            time.sleep(1)
    raise Exception("ComfyUI server did not become ready in time")


def snap16(value):
    return max(256, (int(value) // 16) * 16)


def apply_refine(prompt, keys, job_input, seed, default_denoise, base_w=None, base_h=None):
    """Shared refine + skin wiring for both graphs."""
    hyperreal = bool(job_input.get("hyperreal", True))
    skin_detail = bool(job_input.get("skin_detail", True))

    if not hyperreal:
        prompt[keys["save"]]["inputs"]["images"] = [keys["base_image"], 0]
        for n in keys["refine_nodes"] + keys["skin_nodes"]:
            prompt.pop(n, None)
        return

    if base_w and base_h:
        scale = float(job_input.get("refine_scale", 1.5))
        prompt[keys["refine_resize"]]["inputs"]["width"] = snap16(base_w * scale)
        prompt[keys["refine_resize"]]["inputs"]["height"] = snap16(base_h * scale)

    # Flux2Scheduler pins denoise at 1.0 (redraw); BasicScheduler lets us lower it.
    prompt[keys["refine_sched"]]["inputs"]["denoise"] = float(
        job_input.get("refine_denoise", default_denoise)
    )
    prompt[keys["refine_sched"]]["inputs"]["steps"] = int(job_input.get("refine_steps", 8))
    prompt[keys["refine_sampler"]]["inputs"]["eta"] = float(job_input.get("eta", 1.0))
    prompt[keys["refine_sampler"]]["inputs"]["s_noise"] = float(job_input.get("s_noise", 1.0))
    prompt[keys["refine_seed"]]["inputs"]["noise_seed"] = seed + 1

    if not skin_detail:
        prompt[keys["save"]]["inputs"]["images"] = [keys["refine_image"], 0]
        for n in keys["skin_nodes"]:
            prompt.pop(n, None)


def build_edit(job_input, seed):
    """Reference-conditioned edit: each reference is encoded and chained into
    the conditioning through ReferenceLatent, per the official Flux.2 template."""
    prompt = load_workflow(WORKFLOW_EDIT)
    refs = job_input["images"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("'images' must be a non-empty list")
    if len(refs) > 6:
        logger.warning("More than 6 references given; using the first 6.")
        refs = refs[:6]

    prompt[EDIT["positive"]]["inputs"]["text"] = job_input["prompt"]
    prompt[EDIT["seed"]]["inputs"]["noise_seed"] = seed
    prompt[EDIT["steps"]]["inputs"]["steps"] = int(job_input.get("steps", 4))
    prompt[EDIT["cfg"]]["inputs"]["cfg"] = float(job_input.get("guidance", 1.0))

    megapixels = float(job_input.get("ref_megapixels", 1.0))

    # First reference reuses the nodes already in the file; the rest are cloned.
    encode_nodes = []
    for i, ref in enumerate(refs):
        name = upload_image(fetch_bytes(ref), f"ref_{uuid.uuid4().hex}.png")
        if i == 0:
            load_id, scale_id, enc_id = EDIT["ref_load"], EDIT["ref_scale"], EDIT["ref_encode"]
            prompt[load_id]["inputs"]["image"] = name
            prompt[scale_id]["inputs"]["megapixels"] = megapixels
        else:
            load_id, scale_id, enc_id = f"10_{i}", f"11_{i}", f"12_{i}"
            prompt[load_id] = {
                "inputs": {"image": name},
                "class_type": "LoadImage",
            }
            prompt[scale_id] = {
                "inputs": {
                    "image": [load_id, 0],
                    "upscale_method": "nearest-exact",
                    "megapixels": megapixels,
                },
                "class_type": "ImageScaleToTotalPixels",
            }
            prompt[enc_id] = {
                "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
                "class_type": "VAEEncode",
            }
        encode_nodes.append(enc_id)

    # Chain one ReferenceLatent per image onto both conditioning branches.
    prompt.pop(EDIT["pos_ref"], None)
    prompt.pop(EDIT["neg_ref"], None)
    pos_src, neg_src = ["20", 0], ["21", 0]
    for i, enc in enumerate(encode_nodes):
        pid, nid = f"30_{i}", f"31_{i}"
        prompt[pid] = {
            "inputs": {"conditioning": pos_src, "latent": [enc, 0]},
            "class_type": "ReferenceLatent",
        }
        prompt[nid] = {
            "inputs": {"conditioning": neg_src, "latent": [enc, 0]},
            "class_type": "ReferenceLatent",
        }
        pos_src, neg_src = [pid, 0], [nid, 0]
    prompt[EDIT["guider"]]["inputs"]["positive"] = pos_src
    prompt[EDIT["guider"]]["inputs"]["negative"] = neg_src

    # Output size follows the first reference, so the refine resize is driven by
    # the requested scale rather than a known base width/height.
    scale = float(job_input.get("refine_scale", 1.25))
    prompt[EDIT["refine_resize"]]["inputs"]["width"] = snap16(
        int(job_input.get("width", 1024)) * scale
    )
    prompt[EDIT["refine_resize"]]["inputs"]["height"] = snap16(
        int(job_input.get("height", 1024)) * scale
    )

    # Identity drifts fast in a refine pass, so default lower than for t2i.
    apply_refine(prompt, EDIT, job_input, seed, default_denoise=0.30)
    return prompt


def build_t2i(job_input, seed):
    prompt = load_workflow(WORKFLOW_T2I)
    width = snap16(job_input.get("width", 1024))
    height = snap16(job_input.get("height", 1024))
    hyperreal = bool(job_input.get("hyperreal", True))

    positive = job_input["prompt"]
    if hyperreal and job_input.get("detail_suffix", True):
        positive = f"{positive}, {DETAIL_SUFFIX}"

    prompt[T2I["positive"]]["inputs"]["text"] = positive
    prompt[T2I["negative"]]["inputs"]["text"] = job_input.get("negative_prompt", DEFAULT_NEGATIVE)
    prompt[T2I["seed"]]["inputs"]["noise_seed"] = seed
    prompt[T2I["width"]]["inputs"]["value"] = width
    prompt[T2I["height"]]["inputs"]["value"] = height
    prompt[T2I["steps"]]["inputs"]["steps"] = int(job_input.get("steps", 4))
    prompt[T2I["cfg"]]["inputs"]["cfg"] = float(job_input.get("guidance", 1.0))

    # NAG restores negative prompting on the distilled model (cfg is pinned at 1).
    prompt[T2I["nag"]]["inputs"]["nag_scale"] = float(job_input.get("nag_scale", 5.0))
    prompt[T2I["nag"]]["inputs"]["nag_alpha"] = float(job_input.get("nag_alpha", 0.5))
    prompt[T2I["nag"]]["inputs"]["nag_tau"] = float(job_input.get("nag_tau", 1.5))

    apply_refine(prompt, T2I, job_input, seed, default_denoise=0.45,
                 base_w=width, base_h=height)
    return prompt


def handler(job):
    job_input = job.get("input", {})
    n_refs = len(job_input.get("images") or [])
    logger.info(
        "Received job: mode=%s refs=%d prompt=%.80s",
        "edit" if n_refs else "t2i", n_refs, job_input.get("prompt", ""),
    )

    if "prompt" not in job_input:
        return {"error": "'prompt' is required"}

    seed = int(job_input.get("seed", 0))
    wait_for_server()

    try:
        prompt = build_edit(job_input, seed) if n_refs else build_t2i(job_input, seed)
    except Exception as exc:
        logger.exception("Failed to build the workflow")
        return {"error": f"workflow build failed: {exc}"}

    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}:8188/ws?clientId={client_id}")
    try:
        images = get_images(ws, prompt)
    except Exception as exc:
        logger.exception("Generation failed")
        return {"error": str(exc)}
    finally:
        ws.close()

    if not images:
        return {"error": "No image was produced"}

    return {"image": images[0]}


runpod.serverless.start({"handler": handler})
