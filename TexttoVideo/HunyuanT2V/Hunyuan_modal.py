#Date : 12 Feb 2025
#author : Avishek Paul

# # Text-to-video generation with Hunyuan

# This example demonstrates how to run the Hunyuan text to video generation model by Tencent on Modal.

# Note that the Hunyuan text to video model, at time of writing, requires several minutes on one A100-80GB to produce 
# a high-quality clip of even a few seconds. So a single video generation @50 inference steps would cost about $5 for each video.

# There are several optimization available that can bring down the time further.
# Keep your eyes peeled for improved efficiency
# as the open source community works on this new model.

# ## Setting up the environment for Hunyuan and downloading the model weights

import string
import time
from pathlib import Path
import os
import modal
import requests
import base64
app = modal.App()

# ## Saving outputs

# On Modal, we save large or expensive-to-compute data to
# [distributed Volumes](https://modal.com/docs/guide/volumes)
# We'll use this for saving our weights, as well as our video outputs.

VOLUME_NAME = "Hunyuan-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")  # remote path for saving video outputs

MODEL_VOLUME_NAME = "Hunyuan-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")  # remote path for saving model weights

MINUTES = 60
HOURS = 60 * MINUTES
HF_MODEL_PATH = "tencent/HunyuanVideo"

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

output_log_save_path = f"{OUTPUTS_PATH}/"+'output.log'
# For weights installation instruction https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md
# Note book script https://colab.research.google.com/github/camenduru/hunyuan-video-jupyter/blob/main/hunyuan_video_jupyter.ipynb

## Create image
hunyuan_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install( 
        "transformers",
        "numpy",
        "torch",
        "diffusers",
        "accelerate",
        "sentencepiece",
        "peft",
        "ninja",
        "loguru",
        "wheel",
        "torchvision",
        "GitPython",
        "huggingface_hub[cli]",
        "hf_transfer",
    ).run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git aria2",
        "pip install -v -U git+https://github.com/huggingface/diffusers.git@main#egg=diffusers",
        "pip install -v -U git+https://github.com/huggingface/accelerate.git@main#egg=accelerate",
        "pip install flash-attn --no-build-isolation",
        "pip install xfuser",
    ).env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/models",
        }
    )
)

image_cpu = modal.Image.debian_slim().pip_install("fastapi", "websockets", "requests")
# ## Downloading the model

# We download the model weights into Volume cache to speed up cold starts.

# This download takes five minutes or more, depending on traffic
# and network speed.

# If you want to launch the download first,
# before running the rest of the code,
# use the following command from the folder containing this file:

# ```bash
# modal run --detach Hunyuan_modal::download_model
# ```

# The `--detach` flag ensures the download will continue
# even if you close your terminal or shut down your computer
# while it's running.
with image_cpu.imports():
    from modal.functions import FunctionCall

with hunyuan_image.imports():
    import os
    import base64
    import json

@app.function(
    image=image_cpu,
    volumes={
        MODEL_PATH: model,
    },
    # gpu=GPU_CONFIG,#modal.gpu.H100(count=1),
    timeout=20 * MINUTES,
)
def download_model():
    # uses HF_HOME to point download to the model volume
    import os
    from huggingface_hub import snapshot_download
    import transformers
    from git import Repo
    import subprocess

    def find_folders(target_folder_name, root_dir):
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for dirname in dirnames:
                if target_folder_name in dirname:
                    print(os.path.join(dirpath, dirname), "already exists")
                    return os.path.join(dirpath, dirname)
                
        return None


    print(os.getcwd())
    if not os.path.exists(MODEL_PATH / "Hunyuan"):
        Repo.clone_from("https://github.com/Tencent/HunyuanVideo.git", MODEL_PATH / "Hunyuan")
    else:
        print(MODEL_PATH / "Hunyuan"," already exists")
    
    if not find_folders("hunyuan-video-t2v-720p", MODEL_PATH / "Hunyuan/ckpts"):
        print("Downloading model weights for hunyuan 720p")
        download_path = MODEL_PATH / "Hunyuan/ckpts"
        # subprocess.run(["huggingface-cli", "download", HF_MODEL_PATH, "--local-dir", download_path])  # Alternative way of downloading the model
        snapshot_download(
            HF_MODEL_PATH,
            local_dir=download_path,
        )

    if not find_folders("llava-llama-3-8b-v1_1-transformers", MODEL_PATH / "Hunyuan/ckpts"):
        print("Downloading model weights for hunyuan llava-llama-3-8b")
        download_path = MODEL_PATH / "Hunyuan/ckpts/llava-llama-3-8b-v1_1-transformers"
        # subprocess.run(["huggingface-cli", "download", HF_MODEL_PATH, "--local-dir", download_path])
        snapshot_download(
            "xtuner/llava-llama-3-8b-v1_1-transformers",
            local_dir=download_path,
        )

    if not find_folders("clip-vit-large-patch14", MODEL_PATH / "Hunyuan/ckpts"):
        print("Downloading model weights for clip-vit-large-patch14")
        download_path = MODEL_PATH / "Hunyuan/ckpts/text_encoder_2"
        # subprocess.run(["huggingface-cli", "download", HF_MODEL_PATH, "--local-dir", download_path])
        snapshot_download(
            "openai/clip-vit-large-patch14",
            local_dir=download_path,
        )
    
    subprocess.run(f"cd {MODEL_PATH / 'Hunyuan'}", shell=True)
    print(os.getcwd())
    text_encoder_output_path = MODEL_PATH / "Hunyuan/ckpts/text_encoder"

    # Build the command string
    command = f"python {MODEL_PATH / 'Hunyuan/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py'} " \
            f"--input_dir {MODEL_PATH / 'Hunyuan/ckpts/llava-llama-3-8b-v1_1-transformers'} " \
            f"--output_dir {text_encoder_output_path}"

    # Run the command using subprocess
    subprocess.run(command, shell=True)

    # otherwise, this happens on first inference
    transformers.utils.move_cache()

def print_folder_contents(folder_path):
    for root, dirs, files in os.walk(folder_path):
        print(f"Current directory: {root}")
        
        # List directories
        print("Subdirectories:")
        for dir_name in dirs:
            print(f"  {dir_name}")
        
        # List files
        print("Files:")
        for file_name in files:
            print(f"  {file_name}")

def delete_log_file(output_log_save_path):
    if os.path.exists(output_log_save_path):
        # If the file exists, delete it
        os.remove(output_log_save_path)
        print(f"{output_log_save_path} has been deleted.")
    else:
        print(f"{output_log_save_path} does not exist.")

# Define a custom logging stream handler
class LogStreamHandler:
    def __init__(self, logger, log_capture_string, level):
        self.logger = logger
        self.level = level
        self.log_capture_string = log_capture_string

    def write(self, message):
        if message.strip():  # Ignore empty messages (e.g., extra newlines)
            self.logger.log(self.level, message.strip())
            self.log_capture_string.write(message.strip() + '\n')  # Write to the capture string
            outputs.commit()
    def flush(self):
        pass  # Nothing to flush for logging

# ## Setting up our Hunyuan class

# We'll use the `@cls` decorator to define a [Modal Class](https://modal.com/docs/guide/lifecycle-functions)
# which we use to control the lifecycle of our cloud container.#
# We configure it to use our image, the distributed volume, and GPU as configured earlier.
@app.cls(
    image=hunyuan_image,
    volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
        MODEL_PATH: model,
    },
    gpu=GPU_CONFIG,
    timeout=70 * MINUTES,
    container_idle_timeout=10 * MINUTES,
)
class Hunyuan_text2vid:
    @modal.enter()
    def load_model(self):
        os.chdir(MODEL_PATH / "Hunyuan")
        print(os.getcwd())
        import io
        import sys
        sys.path.append("/models/Hunyuan")
        import logging
        # from hyvideo.config import parse_args
        from hyvideo.inference import HunyuanVideoSampler
        from hyvideo.config import sanity_check_args
        import sys
        import argparse

        MODEL_BASE = os.path.join(os.getcwd(), "ckpts")
        args_dict = {
            'model': "HYVideo-T/2-cfgdistill", # choices = 'HYVideo-T/2', 'HYVideo-T/2-cfgdistill'
            'latent_channels': 16,
            'precision': "bf16", # PRECISIONS = {"fp32", "fp16", "bf16"}
            'rope_theta': 256,
            'vae': "884-16c-hy",
            'vae_precision': "fp16",
            'vae_tiling': True,
            'text_encoder': "llm", #  "llm", "clipL"
            'text_encoder_precision': "fp16",
            'text_states_dim': 4096, 
            'text_len': 256,
            'tokenizer': "llm", 
            'prompt_template' : "dit-llm-encode",
            'prompt_template_video': "dit-llm-encode-video",
            'hidden_state_skip_layer': 2,
            'apply_final_norm': False,
            'text_encoder_2': "clipL",
            'text_encoder_precision_2': "fp16",
            'text_states_dim_2' : 768,
            'tokenizer_2' : "clipL",
            'text_len_2' : 77,
            'denoise_type': 'flow',
            'flow_shift': 7.0,
            'flow_reverse': True,
            'flow_solver': 'euler',
            'use_linear_quadratic_schedule': False,
            'linear_schedule_end': 25,
            'model_base': "ckpts",
            'dit_weight': f"{MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt", #f"{MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt", 
            'model_resolution': '540p', # ["540p", "720p"]
            'load_key': "module",
            'use_cpu_offload': False,
            'batch_size': 1,
            'infer_steps': 20,
            'disable_autocast': False,
            'save_path': './results',
            'save_path_suffix': "",
            'name_suffix': "",
            'num_videos': 1,
            'video_size': (720, 1280),
            'video_length': 129,
            'prompt': "a cat is running, realistic.",
            'seed_type': 'auto',
            'seed': None,
            'neg_prompt': None,
            'cfg_scale': 1.0,
            'embedded_cfg_scale': 6.0,
            'use_fp8': False,
            'reproduce': False,
            'ulysses_degree': 1,
            'ring_degree': 1,
        }
        args = argparse.Namespace(**args_dict)
        sanity_check_args(args)
        print(args)

        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")
    
        # Create save folder to save the samples
        save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
        if not os.path.exists(args.save_path):
            os.makedirs(save_path, exist_ok=True)

        delete_log_file(output_log_save_path)
         # # Create a StringIO buffer to capture the output in a string
        log_capture_string = io.StringIO()

        original_stderr = sys.stderr

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(output_log_save_path),  # Write to a file
            logging.StreamHandler()  # Optionally print to the console
        ])

        # Redirect sys.stdout and sys.stderr to the logging system
        # sys.stdout = LogStreamHandler(logging, logging.INFO)
        sys.stderr = LogStreamHandler(logging, log_capture_string, logging.INFO)

        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        self.hunyuan_video_sampler = hunyuan_video_sampler

        sys.stderr = original_stderr

        # Get the updated args
        args = hunyuan_video_sampler.args
        self.args =args

        print("demo load function")

    @modal.method()
    def generate(
        self,
        prompt,
        height,
        width,
        negative_prompt="",
        infer_steps=20,
        guidance_scale=4.5,
        embedded_guidance_scale=6.0,
        flow_shift=7.0,
        flow_reverse=True,
        use_cpu_offload=False,
        manual_seed=None,
    ):
        from hyvideo.utils.file_utils import save_videos_grid
        from loguru import logger
        from datetime import datetime
        import base64
        from tqdm import tqdm
        import sys
        import io
        import logging

        self.args.video_size = ( height, width)
        self.args.infer_steps = infer_steps
        self.args.guidance_scale = guidance_scale
        self.args.embedded_guidance_scale = embedded_guidance_scale
        self.args.flow_shift = flow_shift
        self.args.flow_reverse = flow_reverse
        self.args.use_cpu_offload = use_cpu_offload

        if negative_prompt != "":
            self.args.negative_prompt = negative_prompt

        if manual_seed is not None:
            self.args.seed_type = 'fixed'
            self.args.seed = manual_seed
        else:
            self.args.seed_type = 'auto'
            self.args.seed = 42

        # # Create a StringIO buffer to capture the output in a string
        log_capture_string = io.StringIO()

        # fh = open(output_log_save_path, 'w')  # one file for both `stdout` and `stderr`
        original_stderr = sys.stderr

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(output_log_save_path),  # Write to a file
            logging.StreamHandler()  # Optionally print to the console
        ])

        sys.stderr = LogStreamHandler(logging, log_capture_string, logging.INFO)

        results = self.hunyuan_video_sampler.predict(
                prompt=prompt, 
                height=self.args.video_size[0],
                width=self.args.video_size[1],
                video_length=self.args.video_length,
                seed=self.args.seed,
                negative_prompt=self.args.neg_prompt,
                infer_steps=self.args.infer_steps,
                guidance_scale=self.args.cfg_scale,
                num_videos_per_prompt=self.args.num_videos,
                flow_shift=self.args.flow_shift,
                batch_size=self.args.batch_size,
                embedded_guidance_scale=self.args.embedded_cfg_scale
            )
        
        sys.stderr = original_stderr

        samples = results['samples']

        mp4_name = slugify(prompt)

        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            # time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{OUTPUTS_PATH}/"+mp4_name
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')
        
        outputs.commit()

        with open(save_path, "rb") as video_file:
            video_bytes = video_file.read()

        delete_log_file(output_log_save_path)

        return base64.b64encode(video_bytes).decode('utf-8')    
    
@app.function(image=image_cpu,
    volumes={
        MODEL_PATH: model,
    },
    timeout=20 * MINUTES)
@modal.web_endpoint(docs=True)
def web_inference( 
        prompt: str,
        height: int = 720,
        width: int = 1280,
        infer_steps: int = 50,
        guidance_scale: float = 4.5,
        embedded_guidance_scale: float = 6.0,
        flow_shift: float = 7.0,
        flow_reverse: bool = True,
        # num_videos_per_prompt: int = 1,
        manual_seed: int = 42,
        # uuid_sent: str = None
    ): 
        function_call = Hunyuan_text2vid().generate.spawn(prompt,
                height=height,
                width=width,
                infer_steps=infer_steps, 
                guidance_scale=guidance_scale,
                embedded_guidance_scale=embedded_guidance_scale,
                flow_shift=flow_shift,
                flow_reverse=flow_reverse,
                # num_videos_per_prompt=num_videos_per_prompt,
                manual_seed=manual_seed,)

        return {"function_call_id": function_call.object_id}

@app.function(image=image_cpu,
    volumes={
        MODEL_PATH: model,
    },
    timeout=20 * MINUTES)
@modal.web_endpoint(docs=True)
def cancel_call(function_call_id: str):
    FunctionCall.from_id(function_call_id).cancel()
    return {"cancelled": True}

@app.function(image=image_cpu,
    volumes={
        MODEL_PATH: model,
    },
    timeout=20 * MINUTES)
@modal.web_endpoint(docs=True)
def get_result(function_call_id: str):
    return FunctionCall.from_id(function_call_id).get()

# ## Running Hunyuan inference

# We can trigger Mochi inference from our local machine by running the code in
# the local entrypoint below.

# It ensures the model is downloaded to a remote volume,
# spins up a new replica to generate a video, also saved remotely,
# and then downloads the video to the local machine.

# You can trigger it with:
# ```bash
# modal run --detach mochi
# ```

# Using these flags, you can tweak your generation from the command line:
# ```bash
# modal run --detach mochi --prompt="a cat playing drums in a jazz ensemble" --num-inference-steps=64
# ```

@app.local_entrypoint()
def main(
    prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
):
    # prompt = "An ultra wide drone footage from top of a mountain, at the bottom primitive humans are looking down from the mountain top and the background has a the sunset, cinematic view"

    start_url = web_inference.web_url
    result_url = get_result.web_url
    cancel_url = cancel_call.web_url

    print("Start url ", start_url)
    print("Result url ", result_url)
    print("Cancel url ", cancel_url)

    params = {
        'prompt': prompt,
        'infer_steps': 2,
        'guidance_scale': 6.0,
        "embedded_guidance_scale": 6.0,
        "flow_shift": 17.0,
        "flow_reverse": True,
        'num_videos_per_prompt': 1,
        'manual_seed': 6789,
    }
    # Define the headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Perform the GET request
    response_function_call_id = requests.get(start_url, headers=headers, params=params)
    function_call_id = json.loads(response_function_call_id.content.decode('utf-8'))['function_call_id']
    print("Function_call_id :", function_call_id)

    response_result_url = requests.get(result_url, headers=headers, params={"function_call_id": function_call_id})
        
    # Check if the request was successful
    if response_result_url.status_code == 200:

        output_vid_base64 = response_result_url.content
        video_data = base64.b64decode(output_vid_base64)
        mp4_name = slugify(prompt)
        print(f"Saving it to {mp4_name}")
        with open(mp4_name, "wb") as output_file:
            output_file.write(video_data)
    else:
        print(f'Error: {response_result_url.status_code}')
        print(response_result_url.text)
# ## Addenda

# The remainder of the code in this file is utility code.


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # since filenames can't be longer than 255 characters
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name