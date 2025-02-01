### Basic setup
from pathlib import Path
import modal
import os

MODEL_VOLUME_NAME = "DeepSeek-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models") 

MINUTES = 60
HOURS = 60 * MINUTES

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = os.environ.get("GPU_COUNT", 2)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

HF_MODEL_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
selected_model = HF_MODEL_PATH.split('/')[1]

deep_seek_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install( 
        "transformers==4.48.1",
        "torch",
        "huggingface_hub[cli]",
        "lmdeploy==0.7.0.post2",
    ).run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git",
        # "git clone -b support-dsv3 https://github.com/InternLM/lmdeploy.git && cd lmdeploy && export TORCH_USE_CUDA_DSA=1 && pip install -e .",
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
# modal run Deepseekr1_image_lmdeploy.py::download_model_to_image --hfrepoid deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

## Once the model has been downloaded after setting the selected model the code can be executed

@app.function(
    image=deep_seek_image,
    volumes={
        MODEL_PATH: model,
    },
    timeout=20 * MINUTES,
)
def download_model_to_image(hfrepoid):
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
        print(f"Downloading repo '{hfrepoid}' ")
        download_path = MODEL_PATH / hfrepoid.split('/')[1]
        snapshot_download(
                hfrepoid,
                local_dir=download_path,
                # revision=MODEL_REVISION,
                # ignore_patterns=["*.pt", "*.bin"],
            )
        print("Download finished")
    else:
        print(f"Repository '{hfrepoid}' does not exist on Hugging Face Hub.")    
    
    model.commit()

@app.cls(cpu=8.0,
         memory=32768,
         volumes={
            MODEL_PATH: model,
            },
            gpu=GPU_CONFIG,
         allow_concurrent_inputs=100,
         timeout=10 * MINUTES,
         image=deep_seek_image)
class DeepSeekR1:

    @modal.enter()
    def load_model(self):

        model.reload()
        
        model_dir = os.path.expanduser(os.path.join(MODEL_PATH, selected_model))

        from lmdeploy import pipeline, PytorchEngineConfig, ChatTemplateConfig

        pipe = pipeline(model_path = model_dir, 
                        chat_template_config = ChatTemplateConfig(model_name='deepseek'),
                        backend_config=PytorchEngineConfig(tp=int(GPU_COUNT)))
        self.pipe = pipe
        print("Model Loading Finished")


    # @modal.method()
    def _inference(self, 
                   prompt,
                   max_new_tokens=1024,
                    top_p=0.8,
                    top_k=40,
                    temperature=0.6,
                    do_sample=True,
                    manual_seed=None):
        print("Inside Inference Function")

        from lmdeploy import GenerationConfig
        import random
        print("Prompt is ", prompt)
        if manual_seed is None:
            manual_seed = random.randint(1, 999999)

        gen_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            do_sample=do_sample,
            random_seed=manual_seed
            )
        
        outputs = self.pipe(prompt, gen_config=gen_config)

        return outputs
    
    @modal.method()
    def inference(self, 
                  prompt, ):
        return self._inference(
            prompt,
        )
    
    @modal.web_endpoint(docs=True)
    def web_inference(
        self, 
        prompt: str,
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature:float = 0.6,
        do_sample: bool = True,
        manual_seed: int = 42,
    ):
        print("Prompt is ", prompt)

        content = self._inference(
                prompt,
                max_new_tokens=max_new_tokens, 
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                do_sample=do_sample,
                manual_seed=manual_seed,
            )
        
        print(content)

        # response_list = []
        # for o in content:
        #     generated_text = o.text.replace('</think>', '')
        #     response_list.append(generated_text)
        # print(response_list)
        return content
    # @modal.exit()
    # def close_connection(self):
    #     self.connection.close()

@app.local_entrypoint()
def main():
    import requests
    import json
    
    model = DeepSeekR1()

    messages_list = [
        [{"role": "user", "content": "Who are you?"}],
        [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V3 adopts innovative architectures to guarantee economical training and efficient inference."}],
        [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
    ]

    url = model.web_inference.web_url
    print(url)

    for each_message in messages_list:
        print(each_message)
        params = {
            'prompt': json.dumps(each_message),
            'max_new_tokens': 1024, 
            'top_p': 0.6,
            'top_k': 40,
            'temperature': 0.2,
            'do_sample': True,
            'manual_seed': 1987,
        }
        # Define the headers
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # Perform the GET request
        response = requests.get(url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            response_dict = json.loads(response.text)['text']
            # for o in response:
            #     generated_text = o.text.replace('</think>', '')
            print(response_dict)
        else:
            print(f'Error: {response.status_code}')
            print(response.text)
