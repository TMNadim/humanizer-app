import os, subprocess
import modal

MODEL_REPO = "jialinyyzz/humanizer"
MODEL_FILE = "humanizer-12b-Q8_0.gguf"
MODELS = "/models"

vol = modal.Volume.from_name("humanizer-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("humanizer", image=image)


@app.function(volumes={MODELS: vol}, timeout=3600)
def download():
    from huggingface_hub import hf_hub_download
    hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=MODELS)
    hf_hub_download(MODEL_REPO, "prompt_format.json", local_dir=MODELS)
    vol.commit()


@app.function(
    gpu="L4",
    volumes={MODELS: vol},
    secrets=[modal.Secret.from_name("humanizer-key")],
    scaledown_window=300,
    timeout=900,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=8080, startup_timeout=600)
def serve():
    subprocess.Popen([
        "/app/llama-server",
        "-m", f"{MODELS}/{MODEL_FILE}",
        "--host", "0.0.0.0", "--port", "8080",
        "-ngl", "99", "-c", "8192", "-np", "1",
        "--api-key", os.environ["HUMANIZER_API_KEY"],
    ])