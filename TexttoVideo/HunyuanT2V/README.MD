# Hunyuan Text to Video

This repository demonstrates how to run the state-of-the-art **Hunyuan Text-to-Video** model on **Modal**, a serverless GPU service with a graphical user interface developed using **Streamlit**. The frontend can be hosted on a server or run locally.

The **Hunyuan model** requires significant GPU memory, approximately **60 GB** to generate a **720p x 1280p** video with **129 frames**. Modal offers various GPU models (A100, A100-80GB, H100, etc.) and allows you to scale up to **8 GPUs** at a cheap hourly rate, with charges only for the duration of use.

The model is available in both **float-point 16** and **float-point 8** variants. It has been tested on the **A100-80GB** model, where typically, **50 inference steps** are needed to generate a high-quality video, with each step taking about **1 minute**. The video generation process can be sped up by adjusting the **ulysses-degree** and **ring-degree**, which parallelize the process and reduce generation time.

## Table of Contents

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Explanation of the Scripts](#explanation-of-the-scripts)
- [Frontend](#frontend)

## Requirements

- Modal account and setup
- Python 3.8+ environment
- Streamlit (for GUI)
- Modal CLI installed

## Getting Started

Follow these steps to get started:

1. **Download the Model**  
   Run the following command to download the model:
   ```bash
   modal run -detach Hunyuan_modal.py::download_model
   ```

This will clone the repository and download the model in the appropriate directory.

2. **Enter a prompt and generate**
  Edit the `app.local_entrypoint` section of `Hunyuan_modal.py` and enter a prompt. Then, run the model to ensure it executes properly:
  ```bash
  modal run Hunyuan_modal.py
  ```

3. **Start endpoint**
   Start the endpoints
   After confirming the model runs correctly, start the api endpoints of the model:
   ```bash
   modal serve Hunyuan_modal.py
   ```

4. **Configure Secrets**
   You will see three URLs printed in the terminal. Add these URLs to the `secrets.toml` file:
   - If running locally, place `secrets.toml` in the `.streamlit` folder.
   - If using Streamlit Community Hub, add the secrets to the Streamlit secrets settings.

5. **Start the Log Endpoint**
   Run the following command to start the log endpoint:
   ```bash
   modal serve terminal_log_modal.py
   ```

6. **Run the GUI locally***
   Start the Streamlit GUI with:
   ```bash
   streamlit run Hunyuan_streamlit_webendpoint.py
   ```
That's it! You can now try out your new **Text-to-Video** setup!

Explanation of how scripts works
The main script is `Hunyuan_modal.py` which contains a function `download_model` to download the model weights, a class to load the model weights and generate videos  based on the given prompt and given parameters. The script also generates three endpoints one for starting the generation process, antoher for getting result and the third and alst one for cancelation of the current process. These endpoints needs to be entered in the `secrets.toml` file. The main script re directs the terminal output to an output.txt file and there is a companion script `terminal_log_modal.py` that reads the output file and sends the content of output file over a websocket to the streamlit frontend. The terminal log script communicates with the streamlit frontend to ensure the connection is alive. 

Finally the frontend is contained in the streamlit app `Hunyuan_streamlit_webendpoint.py` which contains different parameters related to the model ( eg inference steps, flow shift, guidance scale, manual_seed, fp8 or fp16 model selection, prompt etc). During generation the sttramlit app first spawns the process with the given parameters by calling the class, then monitors for getting the result or if the user wants to terminate that can also be done by clicking the cancel button. The video generation process is started on a separate thread while the main thread is kept active and using which the streamlit communicates asynchronously with the terminal_log_script. It gets the model loading section and generation which is displayed on a progress bar and finally the video is displayed along with an option to downlaod the generated video.

Suggestions for video creation

Set Inference Steps to 7~10, select fp8 and enter a prompt and try it out to see if the video is what you expect to be. If not change prompt and/or seed number and try again. Once you have a satisfactory response then increase it to 50 to get a good quality video. Higher value of Embedded Guidance scale will help to follow prompt closely while lower value will make it imaginative. To have a visual idea of how the generation changes with guidance scale, embedded guidance scale and flow shift take a look [here](https://drive.google.com/drive/folders/1KZb5EY0Q9GNqhivOyJPGX5STkGnF3isq)

Also to generate better prompts take some sample prompts from the [paper](https://aivideo.hunyuan.tencent.com/), [project page](https://aivideo.hunyuan.tencent.com/) and use a large language model to ask to devise a similar prompt for the idea you have in mind. 

To do list

Add support for parallel
Improve the prompt
