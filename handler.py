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

# Node ids inside workflow.json (flattened from the ComfyUI subgraph "75:*")
NODE_POSITIVE_PROMPT = "75:74"
NODE_NEGATIVE_PROMPT = "75:67"
NODE_SEED = "75:73"
NODE_WIDTH = "75:68"
NODE_HEIGHT = "75:69"
NODE_STEPS = "75:62"
NODE_CFG = "75:63"


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


def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {job_input}")

    if "prompt" not in job_input:
        return {"error": "'prompt' is required"}

    prompt = load_workflow()

    prompt[NODE_POSITIVE_PROMPT]["inputs"]["text"] = job_input["prompt"]
    prompt[NODE_NEGATIVE_PROMPT]["inputs"]["text"] = job_input.get("negative_prompt", "")
    prompt[NODE_SEED]["inputs"]["noise_seed"] = job_input.get("seed", 0)
    prompt[NODE_WIDTH]["inputs"]["value"] = job_input.get("width", 1024)
    prompt[NODE_HEIGHT]["inputs"]["value"] = job_input.get("height", 1024)
    prompt[NODE_STEPS]["inputs"]["steps"] = job_input.get("steps", 4)
    prompt[NODE_CFG]["inputs"]["cfg"] = job_input.get("guidance", 1.0)

    wait_for_server()

    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}:8188/ws?clientId={client_id}")

    images = get_images(ws, prompt)
    ws.close()

    if not images:
        return {"error": "No image was produced"}

    return {"image": images[0]}


runpod.serverless.start({"handler": handler})
