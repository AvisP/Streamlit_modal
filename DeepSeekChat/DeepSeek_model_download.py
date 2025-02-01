### Basic setup
from pathlib import Path
import modal
import os

MODEL_VOLUME_NAME = "DeepSeek-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models") 

MODEL_VOLUME_NAME = "DeepSeek-unsloth-model"
unsloth_model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
UNSLOTH_MODEL_PATH = Path("/unsloth_models") 

MINUTES = 60
HOURS = 60 * MINUTES

deep_seek_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install( 
        "transformers==4.48.1",
        "torch",
        "huggingface_hub[cli]",
    ).run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git",
        "git clone -b support-dsv3 https://github.com/InternLM/lmdeploy.git && cd lmdeploy && export TORCH_USE_CUDA_DSA=1 && pip install -e .",
    )
    # .run_function(
    #     download_model_to_image,
    # )
)

# Create a modal application
app = modal.App("DeepSeekR1")

with deep_seek_image.imports():
    from pathlib import Path


##3 To download a model onto the volume run the command as an example shown below
# modal run Deepseek_model_download.py::download_model_to_image --hfrepoid deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# modal run Deepseek_model_download.py::download_model_to_image --hfrepoid unsloth/DeepSeek-R1-GGUF --allowpattern "*R1-Q8*"
# modal run Deepseek_model_download.py::download_model_to_image --hfrepoid unsloth/DeepSeek-R1-GGUF --allowpattern "*R1-UD-IQ1*"
## Once the model has been downloaded after setting the selected model the code can be executed

@app.function(
    image=deep_seek_image,
    volumes={
        MODEL_PATH: model,
        UNSLOTH_MODEL_PATH: unsloth_model,
    },
    timeout=120 * MINUTES,
)
def download_model_to_image(hfrepoid, allowpattern=None):
    print(hfrepoid)
    from huggingface_hub import HfApi 
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub import snapshot_download

    def repo_exists(HF_repo_id: str) -> bool:
        try:
            HfApi().model_info(HF_repo_id) 
            return True
        except RepositoryNotFoundError:
            return False
        
    if repo_exists(hfrepoid):
        print(f"Repository '{hfrepoid}' exists on Hugging Face Hub.")

        if not allowpattern:
            print(f"Downloading repo '{hfrepoid}' ")
            download_path = MODEL_PATH / hfrepoid.split('/')[1]
            snapshot_download(
                    hfrepoid,
                    local_dir=download_path,
                    # revision=MODEL_REVISION,
                    # ignore_patterns=["*.pt", "*.bin"],
                )
            print("Download finished ", hfrepoid)
        else:
            print(f"Downloading repo '{hfrepoid}' with allowed pattern '{allowpattern}'")
            download_path = UNSLOTH_MODEL_PATH / hfrepoid.split('/')[1]
            snapshot_download(
                    hfrepoid,
                    local_dir=download_path,
                    # revision=MODEL_REVISION,
                    # ignore_patterns=["*.pt", "*.bin"],
                    allow_patterns = allowpattern
                )
            print("Download finished repo ", hfrepoid, " allowed pattern ", allowpattern)
    else:
        print(f"Repository '{hfrepoid}' does not exist on Hugging Face Hub.")    
    
    model.commit()