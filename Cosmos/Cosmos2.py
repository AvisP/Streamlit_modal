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

VOLUME_NAME = "Cosmos-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")  # remote path for saving video outputs

MODEL_VOLUME_NAME = "Cosmos-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")  # remote path for saving model weights

MINUTES = 60
HOURS = 60 * MINUTES
HF_Cosmos_7B_Text2World = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
HF_Cosmos_14B_Text2World = "nvidia/Cosmos-1.0-Diffusion-14B-Text2World"
HF_Cosmos_7B_Video2World = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"
HF_Cosmos_14B_Video2World = "nvidia/Cosmos-1.0-Diffusion-14B-Video2World"
HF_Cosmos_Tokenizer_CV8x8x8 = "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"
HF_Cosmos_Tokenizer_DV8x16x16 = "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16"
HF_Cosmos_Prompt_Upsampler_12B_Text2World = "nvidia/Cosmos-1.0-Prompt-Upsampler-12B-Text2World" ## needs approval
HF_Pixtral_12B_2409 = "mistralai/Pixtral-12B-2409" ## Needs approval
HF_READ_TOKEN = ### Enter token

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

# For weights installation instruction https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/README.md

## Create image
cosmos_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "attrs==25.1.0",
        "better-profanity==0.7.0",
        "boto3==1.35.99",
        "decord==0.6.0",
        "diffusers==0.32.2",
        "einops==0.8.1",
        "hf_transfer",
        "huggingface-hub==0.29.2",
        "hydra-core==1.3.2",
        "imageio[pyav,ffmpeg]==2.37.0",
        "iopath==0.1.10",
        "ipdb==0.13.13",
        "loguru==0.7.2",
        "mediapy==1.2.2",
        "megatron-core==0.10.0",
        "nltk==3.9.1",
        "numpy==1.26.4",
        "nvidia-ml-py==12.535.133",
        "omegaconf==2.3.0",
        "opencv-python==4.10.0.84",
        "pandas==2.2.3",
        "peft==0.14.0",
        "pillow==11.1.0",
        "protobuf==4.25.3",
        "pynvml==12.0.0",
        "pyyaml==6.0.2",
        "retinaface-py==0.0.2",
        "safetensors==0.5.3",
        "scikit-image==0.25.2",
        "sentencepiece==0.2.0",
        "setuptools==76.0.0",
        "termcolor==2.5.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "tqdm==4.66.5",
        "transformers==4.49.0",
        "wheel"
    ).run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git aria2 clang wget",
        "pip install --upgrade pip",
        "wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb",
        "dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb",
        "cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/",
        "apt-get -y install cudnn-cuda-12",
        "export CPATH=/usr/local/cuda/include:$CPATH",
        "pip install transformer-engine[pytorch]==1.12.0"
    ).env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/models",
        }
    )
    .workdir(MODEL_PATH)
    # .run_function(
    #     download_model,
    # )
)
image_cpu = modal.Image.debian_slim().pip_install("fastapi", 
                                                  "websockets", 
                                                  "requests", 
                                                  "GitPython", 
                                                  "huggingface_hub",
                                                  "huggingface_hub[hf_xet]",
                                                  "torch",
                                                  "safetensors").run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git aria2 clang wget",
        "apt install git",
        "pip install --upgrade pip")

with cosmos_image.imports():
    import os
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video
    import base64


@app.function(
    image=image_cpu,
    volumes={
        MODEL_PATH: model,
    },
    timeout=60 * MINUTES,
    container_idle_timeout=20 * MINUTES
)
def download_model():
    # uses HF_HOME to point download to the model volume
    import os
    # from huggingface_hub import snapshot_download
    # import transformers
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
    print(MODEL_PATH)
    # Define the repository URL and destination directory
    repo_url = "https://github.com/AvisP/cosmos-predict1.git"
    destination_dir = MODEL_PATH / "Cosmos"

    if not os.path.exists(destination_dir):
        print(f"Cloning repository from {repo_url} to {destination_dir}...")
        Repo.clone_from(repo_url, destination_dir)
    else:
        print("Repository already exists locally. Pulling the latest changes...")
        repo = Repo(destination_dir)
        repo.git.pull()
    print("Latest changes pulled...")

    os.chdir(os.path.join(MODEL_PATH, "Cosmos"))
    command = [
    "python",
    "scripts/download_diffusion_checkpoints.py",
    "--model_sizes", "7B", "14B",
    "--model_types", "Text2World", "Video2World",
    "--hf_read_api_key", HF_READ_TOKEN
    ]

    # Run the command
    subprocess.run(command)
    
    # if not find_folders("Cosmos-1.0-Diffusion-7B-Text2World", MODEL_PATH / "Cosmos/checkpoints"):
    #     print("Downloading model weights for Cosmos-1.0-Diffusion-7B-Text2World")
    #     download_path = MODEL_PATH / "Cosmos/checkpoints/Cosmos-1.0-Diffusion-7B-Text2World"
    #     # subprocess.run(["huggingface-cli", "download", HF_MODEL_PATH, "--local-dir", download_path])
    #     snapshot_download(
    #         HF_Cosmos_7B_Text2World,
    #         local_dir=download_path,
    #         # revision=MODEL_REVISION,
    #         # ignore_patterns=["*.pt", "*.bin"],
    #         use_auth_token = HF_READ_TOKEN,
    #     )

@app.function(
    image=image_cpu,
    volumes={
        MODEL_PATH: model,
    },
    timeout=60 * MINUTES,
    container_idle_timeout=20 * MINUTES
)
def install_package():
    # uses HF_HOME to point download to the model volume
    import os
    import subprocess
    # from huggingface_hub import snapshot_download
    # import transformers

    print(os.getcwd())
    print(MODEL_PATH)
    os.chdir(os.path.join(MODEL_PATH, "Cosmos"))
    
    subprocess.run(["python3", "-m", "pip", "install", "-e", ".", "--no-deps"], check=True)

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

def create_args_dict(PROMPT, selected_model, output_save_filename):

    args_dict = {
        "checkpoint_dir": "checkpoints",
        "tokenizer_dir": "Cosmos-Tokenize1-CV8x8x8-720p",
        "video_save_name": output_save_filename,
        "video_save_folder": OUTPUTS_PATH,
        "prompt": PROMPT, 
        "batch_input_path": None,  # No default value provided
        "negative_prompt": (
            "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
            "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
            "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
            "flickering. Overall, the video is of poor quality."
        ),
        "num_steps": 35,
        "guidance": 7.0,
        "num_video_frames": 121,
        "height": 704,
        "width": 1280,
        "fps": 24,
        "seed": 7,
        "num_gpus": 1,
        "disable_prompt_upsampler": True, #False,  # Default is not set to "store_true"
        "offload_diffusion_transformer": False,
        "offload_tokenizer": False,
        "offload_text_encoder_model": False,
        "offload_prompt_upsampler": True,
        "offload_guardrail_models": True,
        "disable_guardrail": True, #False,
        "diffusion_transformer_dir": selected_model,
    }

    return args_dict

# ## Setting up our Hunyuan class


# We'll use the `@cls` decorator to define a [Modal Class](https://modal.com/docs/guide/lifecycle-functions)
# which we use to control the lifecycle of our cloud container.
#
@app.function(
    image=cosmos_image,
    volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
        MODEL_PATH: model,
    },
    gpu=GPU_CONFIG,#modal.gpu.H100(count=1),
    timeout=55 * MINUTES,
    container_idle_timeout=10 * MINUTES,
)
def run_inference():
    import os
    import subprocess
    import sys
    # from huggingface_hub import snapshot_download
    # import transformers

    print(os.getcwd())
    print(MODEL_PATH)
    os.chdir(os.path.join(MODEL_PATH, "Cosmos"))
    print(os.getcwd())
    sys.path.append(os.path.join(MODEL_PATH, "Cosmos"))
    import argparse
    import torch
    from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments, validate_args
    from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionText2WorldGenerationPipeline
    from cosmos_predict1.utils import log, misc
    from cosmos_predict1.utils.io import read_prompts_from_file, save_video

    torch.enable_grad(False)

    selected_model = "Cosmos-Predict1-14B-Text2World"
    # PROMPT = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
    # The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
    # A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
    # suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
    # The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
    # field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
    PROMPT = "Dynamic, roaring ocean wave crashing violently against wooden fishing boats, foam and spray exploding in slow motion, dark storm clouds swirling overhead, mist and rain blowing sideways, Mount Fuji faintly visible in the distance. Traditional Japanese Ukiyo-e art style, vivid indigo and white color palette, woodblock texture, cinematic lighting, hyper-detailed, 4K resolution."
    
    args_dict = create_args_dict(PROMPT, selected_model, "text_world_1")
    args_dict.update({
    "prompt_upsampler_dir": "Cosmos-UpsamplePrompt1-12B-Text2World",
    "word_limit_to_skip_upsampler": 250})        
    args = argparse.Namespace(**args_dict)
    print(args)

    misc.set_random_seed(args.seed)
    inference_type = "text2world"
    validate_args(args, inference_type)

    # Initialize text2world generation model pipeline
    pipeline = DiffusionText2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt}]

    os.makedirs(args.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None:
            log.critical("Prompt is missing, skipping world generation.")
            continue

        # Generate video
        generated_output = pipeline.generate(current_prompt, args.negative_prompt, args.word_limit_to_skip_upsampler)
        if generated_output is None:
            log.critical("Guardrail blocked text2world generation.")
            continue
        video, prompt = generated_output

        if args.batch_input_path:
            video_save_path = os.path.join(args.video_save_folder, f"{i}.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.txt")

        # Save video
        save_video(
            video=video,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))

        log.info(f"Saved video to {video_save_path}")
        log.info(f"Saved prompt to {prompt_save_path}")

@app.function(
    image=cosmos_image,
    volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
        MODEL_PATH: model,
    },
    gpu=GPU_CONFIG,#modal.gpu.H100(count=1),
    timeout=55 * MINUTES,
    container_idle_timeout=10 * MINUTES,
)
def run_video_inference():
    import os
    import sys
    # from huggingface_hub import snapshot_download

    print(os.getcwd())
    print(MODEL_PATH)
    os.chdir(os.path.join(MODEL_PATH, "Cosmos"))
    print(os.getcwd())
    sys.path.append(os.path.join(MODEL_PATH, "Cosmos"))
    import argparse
    import torch
    from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments, check_input_frames, validate_args
    from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline
    from cosmos_predict1.utils import log, misc
    from cosmos_predict1.utils.io import read_prompts_from_file, save_video

    torch.enable_grad(False)

    selected_model = "Cosmos-Predict1-14B-Video2World"
    # PROMPT = "A sophisticated robotic assembly line features two robotics arms working in tandem. The left arm, sleek and black, \
    #      is equpped with a precision griper, delicately handiling a rectangular package labelled 'Kawada'. The right arm, robutst \
    #     and white is similarly outfitted, poised over another package. Both arms are mounted on a metallic base, surrounded by a \
    #     grid-like conveyor system. The background reveals a clean industrial settting with a focus on automation and efficiency.  \
    #     Bright, even lighting highlights the mechanical details, casting subtle reflections on the metallic surfaces. The scene remains \
    #     static, emphasizing the precision and coordination of the robotics arms as they perform their tasks with meticulous accuracy. "
    PROMPT = "Dynamic, roaring ocean wave crashing violently against wooden fishing boats, foam and spray exploding in slow motion, dark storm clouds swirling overhead, mist and rain blowing sideways, Mount Fuji faintly visible in the distance. Traditional Japanese Ukiyo-e art style, vivid indigo and white color palette, woodblock texture, cinematic lighting, hyper-detailed, 4K resolution."

    args_dict = create_args_dict(PROMPT, selected_model, "Tsunami_2")
    args_dict.update({
    "prompt_upsampler_dir": "Pixtral-12B",  # Replace with your desired key and value
    "input_image_or_video_path": str(OUTPUTS_PATH / "Tsunami_by_hokusai_19th_century.jpg"),
    "num_input_frames": 1})
    args = argparse.Namespace(**args_dict)
    print(args)

    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args, inference_type)

    # Initialize text2world generation model pipeline
    pipeline = DiffusionVideo2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )

    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt, "visual_input": args.input_image_or_video_path}]

    os.makedirs(args.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue
        current_image_or_video_path = input_dict.get("visual_input", None)
        if current_image_or_video_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue

        # Check input frames
        if not check_input_frames(current_image_or_video_path, args.num_input_frames):
            continue

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_or_video_path=current_image_or_video_path,
            negative_prompt=args.negative_prompt,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt = generated_output

        print("Video Save Path ", os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4"))
        if args.batch_input_path:
            video_save_path = os.path.join(args.video_save_folder, f"{i}.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.txt")

        # Save video
        save_video(
            video=video,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))

        log.info(f"Saved video to {video_save_path}")
        log.info(f"Saved prompt to {prompt_save_path}")



@app.cls(
    image=cosmos_image,
    volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
        MODEL_PATH: model,
    },
    gpu=GPU_CONFIG,#modal.gpu.H100(count=1),
    timeout=55 * MINUTES,
    container_idle_timeout=10 * MINUTES,
)
class CosmosModel:
    @modal.enter()
    def load_model(self):
        os.chdir(MODEL_PATH / "Cosmos")
        print(os.getcwd())
        # print_folder_contents(MODEL_PATH / "Hunyuan")
        import sys
        sys.path.append(MODEL_PATH / "Cosmos")
        sys.path.append("/models/Cosmos")
        # print(sys.path)

        import sys
        import argparse
        from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb
        # from hyvideo.modules.models import HUNYUAN_VIDEO_CONFIG
        # from hyvideo.constants import PROMPT_TEMPLATE_ENCODE, PROMPT_TEMPLATE_ENCODE_VIDEO 
        # print(list(HUNYUAN_VIDEO_CONFIG.keys()))

        MODEL_BASE = os.path.join(os.getcwd(), "ckpts")
        print("Cosmos class load model function")

        # models_root_path = Path(args.model_base)
        # if not models_root_path.exists():
        #     raise ValueError(f"`models_root` not exists: {models_root_path}")
    
        # # Create save folder to save the samples
        # save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
        # if not os.path.exists(args.save_path):
        #     os.makedirs(save_path, exist_ok=True)

        # hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        # self.hunyuan_video_sampler = hunyuan_video_sampler

        # # Get the updated args
        # args = hunyuan_video_sampler.args
        # self.args =args

    # @modal.method()
    def generate(
        self,
        # prompt,
        # height,
        # width,
        # negative_prompt="",
        # infer_steps=20,
        # guidance_scale=4.5,
        # embedded_guidance_scale=6.0,
        # flow_shift=7.0,
        # flow_reverse=False,
        # use_cpu_offload=False,
        # manual_seed=None,
    ):
        
        import argparse
        import subprocess
        parser = argparse.ArgumentParser(description="Text to world generation demo script")
        print(os.getcwd())
        from cosmos1.models.diffusion.inference.inference_utils import add_common_arguments, validate_args
        # Add common arguments
        add_common_arguments(parser)
        
        # Add text2world specific arguments
        parser.add_argument(
            "--diffusion_transformer_dir",
            type=str,
            default="Cosmos-1.0-Diffusion-7B-Text2World",
            help="DiT model weights directory name relative to checkpoint_dir",
            choices=[
                "Cosmos-1.0-Diffusion-7B-Text2World",
                "Cosmos-1.0-Diffusion-14B-Text2World",
            ],
        )
        parser.add_argument(
            "--prompt_upsampler_dir",
            type=str,
            default="Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
            help="Prompt upsampler weights directory relative to checkpoint_dir",
        )

        parser.add_argument(
            "--word_limit_to_skip_upsampler",
            type=int,
            default=250,
            help="Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value",
        )

        print(parser.parse_args())
        # results = self.hunyuan_video_sampler.predict(
        #         prompt=prompt, 
        #         height=self.args.video_size[0],
        #         width=self.args.video_size[1],
        #         video_length=self.args.video_length,
        #         seed=self.args.seed,
        #         negative_prompt=self.args.neg_prompt,
        #         infer_steps=self.args.infer_steps,
        #         guidance_scale=self.args.cfg_scale,
        #         num_videos_per_prompt=self.args.num_videos,
        #         flow_shift=self.args.flow_shift,
        #         batch_size=self.args.batch_size,
        #         embedded_guidance_scale=self.args.embedded_cfg_scale
        #     )
        # results = self.hunyuan_video_sampler.predict(
        #         prompt=prompt, 
        #         height=720,
        #         width=1280,
        #         video_length=61,
        #         seed=1234,
        #         # negative_prompt=self.args.neg_prompt,
        #         infer_steps=20,
        #         guidance_scale=1.0,
        #         num_videos_per_prompt=1,
        #         flow_shift=7.0,
        #         batch_size=1,
        #         embedded_guidance_scale=6.0
        #     )
        # samples = results['samples']

        # for i, sample in enumerate(samples):
        #     sample = samples[i].unsqueeze(0)
        #     # time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        #     save_path = f"{OUTPUTS_PATH}/"+mp4_name
        #     save_videos_grid(sample, save_path, fps=24) # if video_length 129 then 24 if 61 then 15
        #     logger.info(f'Sample save to: {save_path}')

       
        

            # Run the command using subprocess.run
        # process = subprocess.run(command, capture_output=True)

        # Check the return code
        try:
            # result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Command executed successfully.")
            # print("Stdout:\n", result.stdout)
            # print("Stderr:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            print(f"Return code: {e.returncode}")
            print(f"Stdout:\n{e.stdout}")
            print(f"Stderr:\n{e.stderr}")
        except FileNotFoundError:
            print(f"Error: Command not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        outputs.commit()

        # with open(save_path, "rb") as video_file:
        #     video_bytes = video_file.read()

        return True #base64.b64encode(video_bytes).decode('utf-8')
    
    @modal.web_endpoint(docs=True)
    def web_inference(
        self, 
        prompt: str,
        height: int = 720,
        width: int = 1280,
        infer_steps: int = 50,
        guidance_scale: float = 4.5,
        embedded_guidance_scale: float = 6.0,
        flow_shift: float = 7.0,
        flow_reverse: bool = False,
        # num_videos_per_prompt: int = 1,
        manual_seed: int = 42,
        # uuid_sent: str = None
    ):
        from fastapi import Response

        content = self.generate(
                # prompt,
                # height=height,
                # width=width,
                # infer_steps=infer_steps, 
                # guidance_scale=guidance_scale,
                # embedded_guidance_scale=embedded_guidance_scale,
                # flow_shift=flow_shift,
                # flow_reverse=flow_reverse,
                # # num_videos_per_prompt=num_videos_per_prompt,
                # manual_seed=manual_seed,
                # # uuid_sent = uuid_sent,
            )
        return Response(
            content=content,
            # media_type="video/mp4",
        )

    

@app.local_entrypoint()
def main(
    prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
):

    cosmos = CosmosModel()
    # cosmos.generate.remote()
    # prompt = "A cat walks on the grass, realistic style."
    # print(prompt)
    # mp4_name = hunyuan_text2vid.generate.remote(
    #                             prompt,
    #                             height=720,
    #                             width=1280,
    #                             negative_prompt="",
    #                             infer_steps=20,
    #                             guidance_scale=4.5,
    #                             embedded_guidance_scale=6.0,
    #                             flow_shift=7.0,
    #                             flow_reverse=False,
    #                             use_cpu_offload=False,)

    # local_dir = Path("/tmp/")
    # local_dir.mkdir(exist_ok=True, parents=True)
    # local_path = local_dir / mp4_name
    # local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
    # print(f"🍡 video saved locally at {local_path}")
    # print("Entering and Exiting model entry points")

    url = cosmos.web_inference.web_url
    print(url)

    params = {
        'prompt': prompt,
        'infer_steps': 10,
        'guidance_scale': 6.0,
        "embedded_guidance_scale": 6.0,
        "flow_shift": 7.0,
        "flow_reverse": False,
        'num_videos_per_prompt': 1,
        'manual_seed': 2356,
    }
    # Define the headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    # Perform the GET request
    response = requests.get(url, headers=headers, params=params)

    print(response)

#     # Check if the request was successful
#     if response.status_code == 200:

#         output_vid_base64 = response.content
#         video_data = base64.b64decode(output_vid_base64)
#         # output_path ="recieved.mp4"
#         # print(f"Saving it to {output_path}")
#         mp4_name = slugify(prompt)
#         with open(mp4_name, "wb") as output_file:
#             output_file.write(video_data)
#     else:
#         print(f'Error: {response.status_code}')
#         print(response.text)
# # ## Addenda

# The remainder of the code in this file is utility code.


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # since filenames can't be longer than 255 characters
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name