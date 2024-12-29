# ## Basic setup
import subprocess
from pathlib import Path
import requests
import modal
import time
import torch
import os
from uuid import uuid4
# ## Define a container image

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

MINUTES = 60  # seconds

cuda_version = "12.2.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# MODEL_PATH = "THUDM/CogVideoX-5b"
MODEL_PATH = "THUDM/CogVideoX1.5-5B"

volume = modal.Volume.from_name(
    "cogvideo-volume", create_if_missing=True
)
volume_path = (  # the path to the volume from within the container
    Path("/root") / "data"
)
MODEL_DIR = "/model"

def download_model_to_image():
    import transformers
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        # local_dir=MODAL_SAVE_PATH,
        # revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )

    # otherwise, this happens on first inference
    transformers.utils.move_cache()

cogvideo_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{tag}", add_python="3.11"
    ).apt_install(
        "git", "clang"
    )
    .pip_install( 
        "transformers>=4.46.2",  
        "numpy",
        "torch>=2.4.0",
        "sentencepiece>=0.2.0",
        "moviepy==1.0.3",
        "imageio>=2.35.1", 
        "imageio-ffmpeg>=0.5.1",
        "scikit-video>=1.1.11"
    )
    .run_function(  # download the model by running a Python function
        download_model_to_image,
    ).run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6",
        "pip install -v -U git+https://github.com/huggingface/diffusers.git@main#egg=diffusers",
        "pip install -v -U git+https://github.com/huggingface/accelerate.git@main#egg=accelerate"
    )
)

app = modal.App("Cogvideo-text-video")

  # ## Load model and run inference
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@app.cls(cpu=8.0,
        gpu=GPU_CONFIG,
         memory=65536,
         volumes={volume_path: volume},
         timeout=10 * MINUTES,
         container_idle_timeout=10 * MINUTES,
         allow_concurrent_inputs=100,
         image=cogvideo_image)
class Model:

    @modal.enter()
    def enter(self):
        import torch
        from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
        from diffusers.utils import export_to_video
        
        subprocess.run(["nvidia-smi"])
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated()/1024**2)
        print(torch.cuda.memory_reserved()/1024**2)

        # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
        # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
        # function to use Multi GPUs.
        pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        # pipe.to("cuda") # Turn this off if enabling sequation_cpu_offload, slicing and tiling

        # 2. Set Scheduler.
        # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
        # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B and `CogVideoXDPMScheduler` for CogVideoX-5B.
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
        # 3. Enable CPU offload for the model, enable tiling.
        # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
        pipe.enable_sequential_cpu_offload()  # If these are turned on then pipe.to("cuda") should be turned off and vice versa
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()        

        self.pipe = pipe

    def _inference(self, 
                prompt, 
                num_inference_steps, 
                guidance_scale, 
                num_videos_per_prompt, 
                manual_seed,
                uuid_sent=None):
        from diffusers.utils import export_to_video
        import base64
        start = time.monotonic_ns()
        if uuid_sent:
            request_id = uuid_sent
        else:
            request_id = uuid4()
        print(f"Generating response to request {request_id}")
        print(prompt, num_inference_steps, guidance_scale, num_videos_per_prompt, manual_seed)

        if prompt is None:
            prompt = "A cat holding a sign that says hello world"

         # 4. Generate the video frames based on the prompt.
            # `num_frames` is the Number of frames to generate.
            # This is the default value for 6 seconds video and 8 fps,so 48 frames and will plus 1 frame for the first frame.
            # for diffusers `0.30.1` and after version, this should be 49.

        if "CogVideoX-5b" in MODEL_PATH or "CogVideoX1.5-5B" in MODEL_PATH:
            use_dynamic_cfg_flag = True
        else:
            use_dynamic_cfg_flag = False

        video = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=81,  # 49 for CogVideoX and 81 for CogVideoX1.5
            use_dynamic_cfg=use_dynamic_cfg_flag,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
            generator=torch.Generator(device="cuda").manual_seed(manual_seed),  # Set the seed for reproducibility
        ).frames[0]

        output_path = volume_path / "runs" / str(request_id)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file_path =  Path.joinpath(output_path, 'output.mp4')

        # 5. Export the generated frames to a video file. fps must be 8 for original video.
        export_to_video(video, output_file_path, fps=8)
        print(len(video))
        print(type(video[0]))
        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        with open(output_file_path, "rb") as video_file:
            video_bytes = video_file.read()
        return base64.b64encode(video_bytes).decode('utf-8')


    @modal.method()
    def inference(self, 
                  prompt, 
                  num_inference_steps, 
                  guidance_scale, 
                  num_videos_per_prompt, 
                  manual_seed):
        return self._inference(
            prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            num_videos_per_prompt=num_videos_per_prompt, 
            manual_seed=manual_seed
        )#.getvalue()

    @modal.web_endpoint(docs=True)
    def web_inference(
        self, 
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        manual_seed: int = 42,
        uuid_sent: str = None
    ):
        from fastapi import Response
        return Response(
            content=self._inference(
                prompt, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale, 
                num_videos_per_prompt=num_videos_per_prompt, 
                manual_seed=manual_seed,
                uuid_sent = uuid_sent,
            ),
            media_type="video/mp4",
        )


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --help


@app.local_entrypoint()
def main(prompt: str = (
        "A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, "
        "their delicate wings casting shadows on the petals below. In the background, a grand fountain "
        "cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath "
        "the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth "
        "surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace."
    )
    ):
    import base64
    model = Model()

    url = model.web_inference.web_url
    print(url)

    # prompt = "a dramatic scene from a disaster movie. the camera captures buildings collapsing amid explosions and debris flying through the air. people are running in panic, screams filling the chaotic atmosphere. smoke and dust obscure the background, giving a sense of immediate danger. the lighting is dim, with flickering lights adding to the tension. emergency vehicles with flashing lights are trying to navigate the devastation. the video focuses on the disaster's scale and the frantic efforts of people trying to escape."
    params = {
        'prompt': prompt,
        'num_inference_steps': 10,
        'guidance_scale': 6.0,
        'num_videos_per_prompt': 1,
        'manual_seed': 42
    }
    # Define the headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    ### For testing inference class
    # output_vid_base64 = model.inference.remote(prompt, 
    #             num_inference_steps=10, 
    #             guidance_scale=7.0, 
    #             num_videos_per_prompt=1, 
    #             manual_seed=42)
    
    # video_data = base64.b64decode(output_vid_base64)
    # with open('recieved.mp4', "wb") as output_file:
    #     output_file.write(video_data)
    
    # Perform the GET request
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:

        output_vid_base64 = response.content
        video_data = base64.b64decode(output_vid_base64)
        output_path ="recieved.mp4"
        print(f"Saving it to {output_path}")
        with open('recieved.mp4', "wb") as output_file:
            output_file.write(video_data)
    else:
        print(f'Error: {response.status_code}')
        print(response.text)

