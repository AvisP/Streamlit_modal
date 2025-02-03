### Basic setup
from pathlib import Path
import modal
import os

MODEL_VOLUME_NAME = "DeepSeek-unsloth-model"
unsloth_model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
UNSLOTH_MODEL_PATH = Path("/unsloth_models") 

MINUTES = 60
HOURS = 60 * MINUTES

GPU_TYPE = os.environ.get("GPU_TYPE", "A100-80GB")
GPU_COUNT = os.environ.get("GPU_COUNT", 3)
print('GPU count ', GPU_COUNT)
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

deep_seek_image_unsloth = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install( 
        "transformers==4.48.1",
        "torch",
        "huggingface_hub[cli]",
        "gguf"
    ).run_commands(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git",
        "apt-get install build-essential cmake curl libcurl4-openssl-dev -y",
        "git clone https://github.com/ggerganov/llama.cpp",
        "cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON",
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split",
        "cp llama.cpp/build/bin/llama-* llama.cpp"
    )
    # .run_function(
    #     download_model_to_image,
    # )
)

# Create a modal application
app = modal.App("DeepSeekR1")

with deep_seek_image_unsloth.imports():
    from pathlib import Path


##3 To download a model onto the volume run the command as an example shown below
# modal run Deepseekr1_image_lmdeploy.py::download_model_to_image --hfrepoid deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

## Once the model has been downloaded after setting the selected model the code can be executed

# That makes their outputs worth storing.
# In addition to sending them back to clients,
# like our local command line,
# we'll store the results on a Modal Volume for safe-keeping.

results = modal.Volume.from_name("llamacpp-results", create_if_missing=True)
results_dir = "/root/results"

# You can retrieve the results later in a number of ways.

# You can use the Volume CLI:

# ```bash
# modal volume ls llamacpp-results
# ```

# You can attach the Volume to a Modal `shell`
# to poke around in a familiar terminal environment:

# ```bash
# modal shell --volume llamacpp-results
# # then cd into /mnt
# ```

# Or you can access it from any other Python environment
# by using the same `modal.Volume` call as above to instantiate it:

# ```python
# results = modal.Volume.from_name("llamacpp-results")
# print(dir(results))  # show methods
# ```

# ## Running llama.cpp as a Modal Function

# Now, let's put it all together.

# At the top of our `llama_cpp_inference` function,
# we add an `app.function` decorator to attach all of our infrastructure:

# - the `image` with the dependencies
# - the `volumes` with the weights and where we can put outputs
# - the `gpu` we want, if any

# We also specify a `timeout` after which to cancel the run.

# Inside the function, we call the `llama.cpp` CLI
# with `subprocess.Popen`. This requires a bit of extra ceremony
# because we want to both show the output as we run
# and store the output to save and return to the local caller.
# For details, see the [Addenda section](#addenda) below.

# Alternatively, you might set up an OpenAI-compatible server
# using base `llama.cpp` or its [Python wrapper library](https://github.com/abetlen/llama-cpp-python)
# along with one of [Modal's decorators for web hosting](https://modal.com/docs/guide/webhooks).


@app.cls(cpu=8.0,
    image=deep_seek_image_unsloth,
    volumes={UNSLOTH_MODEL_PATH: unsloth_model, results_dir: results},
    gpu=GPU_CONFIG,
    timeout=30 * MINUTES,
)
class DeepSeekR1_unsloth_llama_cpp:
    # @modal.method()
    def llama_cpp_inference(self,
        model_entrypoint_file: str,
        prompt: str = None,
        n_predict: int = -1,
        args: list[str] = None,
        store_output: bool = True,
        temperature: float = 0.6,
        seed: int = 3407,
        top_k: int = 10,
        top_p: float = 0.9,
        ctx_size: int = 0
    ):
        import subprocess
        from uuid import uuid4

        # if prompt is None:
        #     prompt = DEFAULT_PROMPT  # see end of file
        prompt = "<｜User｜>" + prompt + "<｜Assistant｜>" #"<think>"
        if args is None:
            args = []

        # set layers to "off-load to", aka run on, GPU
        if GPU_CONFIG is not None:
            n_gpu_layers = 9999  # all
        else:
            n_gpu_layers = 0

        if store_output:
            result_id = str(uuid4())
            print(f"🦙 running inference with id:{result_id}")
        # For the parameters
        command = [
            "/llama.cpp/llama-cli",
            "--model",
            os.path.join(UNSLOTH_MODEL_PATH, model_entrypoint_file),
            "--n-gpu-layers", str(n_gpu_layers),
            "--prompt", prompt,
            "--n-predict", str(n_predict),
            "--temp", str(temperature),
            "--seed", str(seed),
            '--top-k', str(top_k),
            '--top-p', str(top_p),
            '--ctx-size', str(ctx_size)
        ] + args

        print("🦙 running commmand:", command, sep="\n\t")
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )

        stdout, stderr = collect_output(p)

        if p.returncode != 0:
            raise subprocess.CalledProcessError(
                p.returncode, command, stdout, stderr
            )

        if store_output:  # save results to a Modal Volume if requested
            print(f"🦙 saving results for {result_id}")
            result_dir = Path(results_dir) / result_id
            result_dir.mkdir(
                parents=True,
            )
            (result_dir / "out.txt").write_text(stdout)
            (result_dir / "err.txt").write_text(stderr)

        return stdout
    
    @modal.method()
    def inference(self, 
                  model_entrypoint_file: str,
                    prompt: str = None,
                    n_predict: int = -1,
                    args: list[str] = None,
                    store_output: bool = True,
                    temperature: float = 0.6,
                    seed: int = 3407,
                    top_k: int = 10,
                    top_p: float = 0.9,
                    ctx_size: int = 8192, ):
        return self.llama_cpp_inference(
            model_entrypoint_file,
            prompt,
            n_predict=n_predict,
            args=args,
            store_output=store_output,
            temperature=temperature,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            ctx_size=ctx_size
        )

    @modal.web_endpoint(docs=True)
    def web_inference(
        self, 
        model_entrypoint_file: str,
        prompt: str = None,
        n_predict: int = -1,
        top_p: float = 0.9,
        top_k: int = 10,
        temperature: float = 0.6,
        store_output: bool = True,
        seed: int = 3407,
        ctx_size: int = 8192,
    ):
        print("Prompt is ", prompt)
        content  = "Test return"
        content = self.llama_cpp_inference(
                model_entrypoint_file, 
                prompt,
                n_predict=n_predict,
                args=['-no-cnv', '--color', '--no-warmup'],
                store_output=store_output,
                temperature=temperature,
                seed=seed,
                top_k = top_k,
                top_p = top_p,
                ctx_size = ctx_size
            )
        
        print(content)

        return content
# # Addenda

# The remainder of this code is less interesting from the perspective
# of running LLM inference on Modal but necessary for the code to run.

# For example, it includes the default "Flappy Bird in Python" prompt included in
# [unsloth's announcement](https://unsloth.ai/blog/deepseekr1-dynamic)
# of their 1.58 bit quantization of DeepSeek-R1.

DEFAULT_PROMPT = """Create a Flappy Bird game in Python. You must include these things:

    You must use pygame.
    The background color should be randomly chosen and is a light shade. Start with a light blue color.
    Pressing SPACE multiple times will accelerate the bird.
    The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.
    Place on the bottom some land colored as dark brown or yellow chosen randomly.
    Make a score shown on the top right side. Increment if you pass pipes and don't hit them.
    Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.
    When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.

The final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section."""


def stream_output(stream, queue, write_stream):
    """Reads lines from a stream and writes to a queue and a write stream."""
    for line in iter(stream.readline, b""):
        line = line.decode("utf-8", errors="replace")
        write_stream.write(line)
        write_stream.flush()
        queue.put(line)
    stream.close()


def collect_output(process):
    """Collect up the stdout and stderr of a process while still streaming it out."""
    import sys
    from queue import Queue
    from threading import Thread

    stdout_queue = Queue()
    stderr_queue = Queue()

    stdout_thread = Thread(
        target=stream_output, args=(process.stdout, stdout_queue, sys.stdout)
    )
    stderr_thread = Thread(
        target=stream_output, args=(process.stderr, stderr_queue, sys.stderr)
    )
    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()
    process.wait()

    stdout_collected = "".join(stdout_queue.queue)
    stderr_collected = "".join(stderr_queue.queue)

    return stdout_collected, stderr_collected

@app.local_entrypoint()
def main():
    import requests
    import json
    import os

    DeepSeekR1_class = DeepSeekR1_unsloth_llama_cpp()

    ##### Web point Check section
    url = DeepSeekR1_class.web_inference.web_url
    print(url)

          # Define the headers
    headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    n_predict = 512  # max number of tokens to predict, -1 is infinite
    
    prompt = 'Which one is greater 9.99 or 9.11?'
    
    model_entrypoint_file = "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf"
    params = {
            'model_entrypoint_file': model_entrypoint_file,
            'prompt': prompt,
            'n_predict': n_predict, 
            'top_p': 0.9,
            'top_k': 10,
            'temperature': 0.2,
            'do_sample': True,
            'seed': 1987,
            'ctx_size': 8192,
            'store_output':True,
        }

    # Perform the GET request
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        print("Recieved response")
        response_dict = response.text.split('</think>').replace('[end of text]', '').replace('</think>', '')
        response_dict.replace(prompt, '')

        print(response_dict)
    else:
        print(f'Error: {response.status_code}')
        print(response.text)
        

    # model = "DeepSeek-R1"  # or "phi-4"
    
    # # model_entrypoint_file = os.path.join('DeepSeek-R1-GGUF', 'DeepSeek-R1-UD-IQ1_M')
    # model_entrypoint_file = "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf"
    # # call out to a `.remote` Function on Modal for inference
    # result = llama_cpp_inference.remote(
    #     model_entrypoint_file,
    #     prompt,
    #     n_predict,
    #     args,
    #     store_output=model.lower() == "deepseek-r1",
    # )
    # output_path = Path("/tmp") / f"llama-cpp-{model}.txt"
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # print(f"🦙 writing response to {output_path}")
    # output_path.write_text(result)
    
    # messages_list = [
    #     [{"role": "user", "content": "Who are you?"}],
    #     [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V3 adopts innovative architectures to guarantee economical training and efficient inference."}],
    #     [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
    # ]

    # url = model.web_inference.web_url
    # print(url)

    # for each_message in messages_list:
    #     print(each_message)
    #     params = {
    #         'prompt': json.dumps(each_message),
    #         'max_new_tokens': 1024, 
    #         'top_p': 0.6,
    #         'top_k': 40,
    #         'temperature': 0.2,
    #         'do_sample': True,
    #         'manual_seed': 1987,
    #     }
  
