Chat UI with Deep Seek Distilled models hosted on Modal server

[![Watch the video](Videos/DeepSeekDistilledScreenshot.jpg)](Videos/DemoDeepSeekDistilled.mp4)

# DeepSeekr1 Modal Setup and Usage Guide

## Prerequisites
**Create an account on Modal**  
   Set up your account and environment by following the instructions [here](https://modal.com/docs/guide).

## Local Execution

### Overview
Each model is contained in a separate Python file. For example, `Deepseekr1_modal_lmdeploy_Llama_8B.py` contains the distilled DeepSeek `Llama_8B` model, and other files follow a similar structure.

### Running Models Locally
To test an individual model, run the following command:
```bash
modal run Deepseekr1_modal_lmdeploy_[MODEL_NAME].py
```

Each file contains a `@app.local_entrypoint` that initializes a message_list.
## Creating an API Endpoint

To create an API endpoint for a model, use the following command:

```bash
 modal serve Deepseekr1_modal_lmdeploy_[MODEL_NAME].py
``` 
Note the url created on the terminal when executing this command which should be put in the `secrets.toml` file in the `url` variable. The `url` should be same irresepective of the model being started.

## Downloading Model weights
The models needs weight that can be easily downloaded using the script `DeepSeek_model_download.py`. The distilled models are stored in a persistent volume mapped to `/models/DeepSeek-model`
Example use to download a model weight 

```bash
modal run Deepseek_model_download.py::download_model_to_image --hfrepoid deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```
You can also connect to a shell by running: 
```bash
model shell Deepseek_model_download.py::download_model_to_image
``` 
Once in the shell, use Linux commands to monitor the files.
In case of issues with downloading, use the shell, start Python, and use the appropriate path along with the `snapshot_download` command from `huggingface_hub` to complete the download.

The modal files use `lmdeploy` python library to load up the DeepSeek models and this is needed on the cliend side as well.

### (Not tested) 
The original `Deepseek R1 671B` can also be run similarly loaded up using the script `Deepseekr1_modal_lmdeploy_670B.py` after having downloaded the weights using the script `DeepSeek_model_download.py`.
However it takes about 10 `A100-80GB` [source](https://github.com/InternLM/lmdeploy/issues/2960) to load up the model in `fp8` format and sepcial permission from `modal` is needed to have access to that many GPUs. However if on any other platform/server there is same amount of GPU resource available then one can try it out. 

## GUI
## Overview
It is recommended to test out the local execution of the individual model files before trying out the GUI

## Running the GUI
The GUI can be started with the command 

```bash
streamlit run streamlit_modal_app_distilled_models.py
```

## Authentication

To authenticate, create a `secrets.toml` file. A sample `secrets.toml` file is included in the repository.

- **For local execution**:  
  Create a `.streamlit` folder in your project directory and place the `secrets.toml` file there.

- **For deployment on Streamlit Community Cloud**:  
  Enter the credentials directly in the Streamlit secrets.

## Using the GUI

Once authentication is successful, you will be presented with a page where you can enter a prompt, set parameters, and submit a query. The `Llama_8B` model will start automatically, so there’s no need to manually run the Modal scripts.

### Notes:
- **Initial Request**:  
  The first user request may take some time to load the model using the `lmdeploy` script. Subsequent requests will be faster. However, the `lmdeploy` engine may shut down after a period of inactivity.

- **Model Switching**:  
  Switching between models may take additional time, especially for larger models with more parameters.

- **Long Chats**:  
  For extended chats, it is recommended to increase the number of new tokens to prevent truncated responses. The models often provide verbose output to indicate their reasoning process.


