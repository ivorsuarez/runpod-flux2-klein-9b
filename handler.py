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

WORKFLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow.json")

# --- Node ids inside workflow.json ---------------------------------------
# base pass
NODE_POSITIVE = "75:74"
NODE_NEGATIVE = "75:67"
NODE_SEED = "75:73"
NODE_WIDTH = "75:68"
NODE_HEIGHT = "75:69"
NODE_STEPS = "75:62"          # Flux2Scheduler
NODE_CFG = "75:63"            # CFGGuider
NODE_NAG = "100"              # NAGuidance
NODE_BASE_IMAGE = "75:65"     # VAEDecode of base pass
# refine pass
NODE_REFINE_RESIZE = "101"    # ImageScale
NODE_REFINE_SCHED = "103"     # BasicScheduler (steps + denoise)
NODE_REFINE_SAMPLER = "104"   # SamplerEulerAncestralCFGPP (eta, s_noise)
NODE_REFINE_SEED = "105"      # RandomNoise
NODE_REFINE_IMAGE = "107"     # VAEDecode of refine pass
# skin pass
NODE_SKIN_IMAGE = "109"       # ImageUpscaleWithModel
NODE_SAVE = "9"

REFINE_NODES = ["101", "102", "103", "104", "105", "106", "107"]
SKIN_NODES = ["108", "109"]

DEFAULT_NEGATIVE = (
    "plastic skin, airbrushed, waxy, smooth skin, cgi, 3d render, illustration, "
    "painting, oversaturated, overexposed, blurry, low detail"
)
# Detail "backbone" appended to the prompt for the refine pass.
DETAIL_SUFFIX = "8K, intricate details"


def load_workflow():
    with open(WORKFLOW_PATH, "r") as f:
        return json.load(f)


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    data = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())


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
    for node_id, node_output in history["outputs"].items():
        if "images" in node_output:
            for image in node_output["images"]:
                image_data = get_image(image["filename"], image["subfolder"], image["type"])
                images_output.append(base64.b64encode(image_data).decode("utf-8"))
    return images_output


def wait_for_server(timeout=180):
    http_url = f"http://{server_address}:8188/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(http_url, timeout=5)
            return
        except Exception:
            time.sleep(1)
    raise Exception("ComfyUI server did not become ready in time")


def snap16(value):
    """Latent grid alignment: round down to a multiple of 16, min 256."""
    return max(256, (int(value) // 16) * 16)


def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {job_input}")

    if "prompt" not in job_input:
        return {"error": "'prompt' is required"}

    prompt_text = job_input["prompt"]
    width = snap16(job_input.get("width", 1024))
    height = snap16(job_input.get("height", 1024))
    seed = int(job_input.get("seed", 0))

    hyperreal = bool(job_input.get("hyperreal", True))
    skin_detail = bool(job_input.get("skin_detail", True))

    prompt = load_workflow()

    # ---- base pass --------------------------------------------------------
    positive = prompt_text
    if hyperreal and job_input.get("detail_suffix", True):
        positive = f"{prompt_text}, {DETAIL_SUFFIX}"

    prompt[NODE_POSITIVE]["inputs"]["text"] = positive
    prompt[NODE_NEGATIVE]["inputs"]["text"] = job_input.get("negative_prompt", DEFAULT_NEGATIVE)
    prompt[NODE_SEED]["inputs"]["noise_seed"] = seed
    prompt[NODE_WIDTH]["inputs"]["value"] = width
    prompt[NODE_HEIGHT]["inputs"]["value"] = height
    prompt[NODE_STEPS]["inputs"]["steps"] = int(job_input.get("steps", 4))
    prompt[NODE_CFG]["inputs"]["cfg"] = float(job_input.get("guidance", 1.0))

    # NAG restores negative prompting on the distilled model (cfg is pinned at 1).
    prompt[NODE_NAG]["inputs"]["nag_scale"] = float(job_input.get("nag_scale", 5.0))
    prompt[NODE_NAG]["inputs"]["nag_alpha"] = float(job_input.get("nag_alpha", 0.5))
    prompt[NODE_NAG]["inputs"]["nag_tau"] = float(job_input.get("nag_tau", 1.5))

    if not hyperreal:
        # Base image only: drop the refine + skin chain.
        prompt[NODE_SAVE]["inputs"]["images"] = [NODE_BASE_IMAGE, 0]
        for node_id in REFINE_NODES + SKIN_NODES:
            prompt.pop(node_id, None)
    else:
        # ---- refine pass --------------------------------------------------
        scale = float(job_input.get("refine_scale", 1.5))
        prompt[NODE_REFINE_RESIZE]["inputs"]["width"] = snap16(width * scale)
        prompt[NODE_REFINE_RESIZE]["inputs"]["height"] = snap16(height * scale)

        # Flux2Scheduler pins denoise at 1.0, which redraws the image instead of
        # refining it — BasicScheduler is used here so denoise can be lowered.
        prompt[NODE_REFINE_SCHED]["inputs"]["denoise"] = float(job_input.get("refine_denoise", 0.45))
        prompt[NODE_REFINE_SCHED]["inputs"]["steps"] = int(job_input.get("refine_steps", 8))
        prompt[NODE_REFINE_SCHED]["inputs"]["scheduler"] = job_input.get("refine_scheduler", "sgm_uniform")

        prompt[NODE_REFINE_SAMPLER]["inputs"]["eta"] = float(job_input.get("eta", 1.0))
        # s_noise above 1.0 adds detail but can speckle; 1.2 is the usual ceiling.
        prompt[NODE_REFINE_SAMPLER]["inputs"]["s_noise"] = float(job_input.get("s_noise", 1.0))
        prompt[NODE_REFINE_SEED]["inputs"]["noise_seed"] = seed + 1

        if not skin_detail:
            prompt[NODE_SAVE]["inputs"]["images"] = [NODE_REFINE_IMAGE, 0]
            for node_id in SKIN_NODES:
                prompt.pop(node_id, None)

    wait_for_server()

    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}:8188/ws?clientId={client_id}")

    images = get_images(ws, prompt)
    ws.close()

    if not images:
        return {"error": "No image was produced"}

    return {"image": images[0]}


runpod.serverless.start({"handler": handler})
