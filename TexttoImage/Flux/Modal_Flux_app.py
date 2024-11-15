### Basic setup
import io
from pathlib import Path
import requests
import modal
import time
import torch
import os
from uuid import uuid4
import base64
# ## Define a container image, install packages

GPU_TYPE = os.environ.get("GPU_TYPE", "A100")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
MINUTES = 60  # seconds
MODEL_PATH = "black-forest-labs/FLUX.1-schnell"
TOKENIZER_PATH = "black-forest-labs/FLUX.1-schnell"

volume = modal.Volume.from_name(
    "Flux-volume", create_if_missing=True
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
        ignore_patterns=["*.pt", "*.bin"],
    )

    # otherwise, this happens on first inference
    transformers.utils.move_cache()

flux_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install( 
        "transformers",
        "numpy",
        "torch",
        "diffusers",
        "accelerate",
        "sentencepiece",
        "peft"
    )
    .run_function(
        download_model_to_image,
    )
)

# Create a modal application
app = modal.App("flux-text-image")

with flux_image.imports():
    import torch
    from diffusers import DiffusionPipeline
    from fastapi import Response
    from fastapi.responses import JSONResponse
    import json

# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@app.cls(cpu=8.0,
        gpu=GPU_CONFIG,
        # gpu=modal.gpu.A10G(), 
         memory=32768,
         volumes={volume_path: volume},
        #  container_idle_timeout=240,
         timeout=5 * MINUTES,
         container_idle_timeout=5 * MINUTES,
         allow_concurrent_inputs=100,
         image=flux_image)
class Model:

    @modal.enter()
    def enter(self):
        import torch
        from diffusers import FluxPipeline
        import subprocess

        # subprocess.run(["nvidia-smi"])
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated()/1024**2)
        print(torch.cuda.memory_reserved()/1024**2)

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        self.pipe = pipe

    def _inference(self, 
                   prompt, 
                   n_steps, 
                   guidance_scale, 
                   max_sequence_length, 
                   manual_seed):

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")
        print(prompt, n_steps, guidance_scale, max_sequence_length, manual_seed)
        
        if prompt is None:
            prompt = "A cat holding a sign that says No prompt found"
        image = self.pipe(
            prompt,
            # negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=n_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(manual_seed)
        ).images[0]

        model_path = volume_path / "runs" / str(request_id)
        model_path.mkdir(parents=True, exist_ok=True)

        image_path =  Path.joinpath(model_path, 'Flux.png')

        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        with open(image_path, "wb") as file:
            file.write(byte_stream.getvalue())

        return byte_stream.getvalue(), request_id

    @modal.method()
    def inference(self, 
                  prompt, 
                  n_steps, 
                  guidance_scale, 
                  max_sequence_length, 
                  manual_seed):
        
        byte_image, request_id =self._inference(
                prompt, 
                n_steps=n_steps, 
                guidance_scale=guidance_scale, 
                max_sequence_length=max_sequence_length, 
                manual_seed=manual_seed)

        return byte_image, request_id

    @modal.web_endpoint(docs=True)
    def web_inference(
        self, 
        prompt: str = 'A default prompt', 
        n_steps: int = 24, 
        guidance_scale: float = 0.8,
        max_sequence_length: int = 256,
        manual_seed: int = None,
        lora_path: str = None,
        lora_weight: str = None,
    ):
        import random
        if not manual_seed:
            manual_seed = random.randint(0, 65535)

        if lora_path and lora_weight:
            print("Lora repo ", lora_path)
            print("Lora file name", lora_weight)
            self.pipe.load_lora_weights(lora_path, weight_name=lora_weight)
        else:
            self.pipe.unload_lora_weights()
            
        byte_image, request_id =self._inference(
                prompt, 
                n_steps=n_steps, 
                guidance_scale=guidance_scale, 
                max_sequence_length=max_sequence_length, 
                manual_seed=manual_seed)
        
        encoded_image = base64.b64encode(byte_image).decode()
        json_request_id = json.dumps({'uuid': str(request_id)})
        return JSONResponse(content={"request_id": json_request_id, "image": encoded_image})


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --help


@app.local_entrypoint()
def main(prompt: str = "Unicorns and leprechauns sign a peace treaty"):
    from PIL import Image
    from io import BytesIO
    import random
    model = Model()

    byte_image, request_id = model.inference.remote(prompt,
                                           n_steps = 5,
                                           guidance_scale=0.8, 
                                            max_sequence_length=256, 
                                            manual_seed=random.randint(0, 999999))

    dir = Path("./Flux-images/"+str(request_id))
    print(dir)
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    image_path =  Path.joinpath(dir, 'Flux.png')

    with open(image_path, "wb") as file:
        file.write(byte_image)

    # output_path = dir / "output.png"
    # print(f"Saving it to {output_path}")
    # with open(output_path, "wb") as f:
    #     f.write(image_bytes)
    # url = model.web_inference.web_url

    # params = {
    #     'prompt': prompt,
    #     'n_steps': 12, 
    #     'guidance_scale': 0.8,
    #     'max_sequence_length': 256,
    #     'manual_seed': 1234,
    # }
    # # Define the headers
    # headers = {
    #     'accept': 'application/json',
    #     'Content-Type': 'application/json'
    # }
    # # Perform the GET request
    # response = requests.get(url, headers=headers, params=params)

    # if response.status_code == 200:
    #     # Print the response content (optional)
    #     data = response.json()  # Parse the JSON response
    #     # print(response.content.image_content)
    #     encoded_image = data['image']
    #     request_id = data['request_id']

    #     image_data = base64.b64decode(encoded_image)
    #     image = Image.open(BytesIO(image_data))

    #     output_path ="output1.png"
    #     print(f"Saving it to {output_path}")
    #     image.save(output_path)
    # else:
    #     print(f'Error: {response.status_code}')
    #     print(response.text)
