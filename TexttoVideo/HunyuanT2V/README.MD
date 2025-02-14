# Hunyuan Text to Video

This repository demonstrates how to run the state-of-the-art **Hunyuan Text-to-Video** model on **Modal**, a serverless GPU service with a graphical user interface developed using **Streamlit**. The frontend can be hosted on a server or run locally.

The **Hunyuan model** requires significant GPU memory, approximately **60 GB** to generate a **720p x 1280p** video with **129 frames**. Modal offers various GPU models (A100, A100-80GB, H100, etc.) and allows you to scale up to **8 GPUs** at a cheap hourly rate, with charges only for the duration of use.

The model is available in both **float-point 16** and **float-point 8** variants. It has been tested on the **A100-80GB** model, where typically, **50 inference steps** are needed to generate a high-quality video, with each step taking about **1 minute**. The video generation process can be sped up by adjusting the **ulysses-degree** and **ring-degree**, which parallelize the process and reduce generation time.

## Table of Contents

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Explanation of the Scripts](#explanation-of-the-scripts)
- [Suggestions](#suggestions-for-video-creation)

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

## Explanation of the Scripts

### Main Script: `Hunyuan_modal.py`

- **`download_model` function**: Downloads the model weights.
- **Class to load model weights and generate videos**: This class handles the video generation process based on the given prompt and parameters.
- **Endpoints**: Three endpoints are created:
  1. **Start generation**: Initiates the video generation process.
  2. **Get result**: Fetches the generated video.
  3. **Cancel process**: Cancels the current video generation process.
  
  These endpoints need to be added to the `secrets.toml` file.

The script redirects terminal output to an `output.txt` file, which is then read by the **log script**.

---

### Companion Script: `terminal_log_modal.py`

- **Reads the output file** (`output.txt`).
- **Sends the contents over a WebSocket** to the Streamlit frontend.
- **Ensures the connection is alive** by communicating with the frontend.

---

### Frontend: `Hunyuan_streamlit_webendpoint.py`

- **Model Parameters**:  
  The frontend contains input fields for various model parameters such as:
  - Inference steps
  - Flow shift
  - Guidance scale
  - Manual seed
  - FP8 or FP16 model selection
  - Prompt input

- **Generation Process**:  
  During generation, the app spawns a process with the provided parameters and monitors for results. If the user decides to cancel the process, this can be done through a cancel button.

- **Asynchronous Communication**:  
  The video generation happens on a separate thread, while the main thread remains active. The Streamlit app communicates asynchronously with the terminal log script, displaying progress and updating the UI with the generated video.

- **Progress Bar**:  
  The app shows a progress bar during the model loading and generation phases. Once complete, the video is displayed along with an option to download it.

## Suggestions for Video Creation

1. **Set Inference Steps**: Start by setting inference steps to **7-10**, select **FP8** model, and enter a prompt. Try it out to see if the generated video matches your expectations. If not, modify the prompt and/or seed number, and try again.

2. **Fine-Tuning**: Once you have a satisfactory result, increase the inference steps to **50** for better video quality.

3. **Embedded Guidance Scale**: 
   - A higher value for the Embedded Guidance scale will help the model follow the prompt more closely.
   - A lower value will make the model more imaginative and creative.

4. **Visual Guide**: To understand how the generation changes with the **Guidance Scale**, **Embedded Guidance Scale**, and **Flow Shift**, refer to the visual guide [here](https://drive.google.com/drive/folders/1KZb5EY0Q9GNqhivOyJPGX5STkGnF3isq).

5. **Improving Prompts**: To generate better prompts, refer to sample prompts from the following sources:
   - [Paper](https://aivideo.hunyuan.tencent.com/)
   - [Project Page](https://aivideo.hunyuan.tencent.com/)
   
   You can also use a large language model to help generate similar prompts based on your idea.

---

## To-Do List

- Add support for **parallel computation**.
- Improve the **prompt** generation process using a **Large Language Model (LLM)**.
- Add an **authentication** feature.

