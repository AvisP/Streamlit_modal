# Hunyuan Text to Video

This repository shows how to run the state of the art [Hunyuan](https://github.com/Tencent/HunyuanVideo) Text to Video model on Modal a serverless GPU service with a Graphical user interface developed with streamlit. The frontend streamlit can be hosted on a server or run locally. The model takes signficant GPU memory to run, based on the provided information it is 60 GB to generate a 720p by 1280p with 129 frames. Modal provides various GPU model (A100, A100-80GB, H100 etc) and as many number needed (8 max) at cheap hourly rate and charged for the duration of use only.

The model is available in float-point 16 and floating point 8 variant. Tested on A100-80GB, typically it needs 50 inference steps to generate good quality video and each step takes about a minute to generate. The model can be sped up by changing the `ulysses-degre` and `ring-degree` that can parallize the video generation process and reduce the video generation time.

Steps to get start
First download the model by running the command `modal run -detach Hunyuan_modal.py::download_model` which should git clone the repo and download the model in the appropriate directory. 

Next enter a prompt in the `app.local_entrypoint` section of `Hunyuan_modal.py` and do a `modal run Hunyuan_modal.py` to see if it executes properly and then execute `modal serve Hunyuan_modal.py`. Also not the different urls of three endoints printed on the terminal that needs to be entered in the `secrets.toml` file that should be placed in a `.streamlit` folder if run locally or put on the secrets if on `streamlit community hub`. Next start the log endpoint by doing `modal serve terminal_log_modal.py` and start the GUI by doing a `streamlit run Hunyuan_streamlit_webpoint.py` locally. And that's it try out your new text to video setup!

Explanation of how scripts works
The main script is `Hunyuan_modal.py` which contains a function `download_model` to download the model weights, a class to load the model weights and generate videos  based on the given prompt and given parameters. The script also generates three endpoints one for starting the generation process, antoher for getting result and the third and alst one for cancelation of the current process. These endpoints needs to be entered in the `secrets.toml` file. The main script re directs the terminal output to an output.txt file and there is a companion script `terminal_log_modal.py` that reads the output file and sends the content of output file over a websocket to the streamlit frontend. The terminal log script communicates with the streamlit frontend to ensure the connection is alive. 

Finally the frontend is contained in the streamlit app `Hunyuan_streamlit_webendpoint.py` which contains different parameters related to the model ( eg inference steps, flow shift, guidance scale, manual_seed, fp8 or fp16 model selection, prompt etc). During generation the sttramlit app first spawns the process with the given parameters by calling the class, then monitors for getting the result or if the user wants to terminate that can also be done by clicking the cancel button. The video generation process is started on a separate thread while the main thread is kept active and using which the streamlit communicates asynchronously with the terminal_log_script. It gets the model loading section and generation which is displayed on a progress bar and finally the video is displayed along with an option to downlaod the generated video.

Suggestions for video creation

Set Inference Steps to 7~10, select fp8 and enter a prompt and try it out to see if the video is what you expect to be. If not change prompt and/or seed number and try again. Once you have a satisfactory response then increase it to 50 to get a good quality video. Higher value of Embedded Guidance scale will help to follow prompt closely while lower value will make it imaginative. To have a visual idea of how the generation changes with guidance scale, embedded guidance scale and flow shift take a look [here](https://drive.google.com/drive/folders/1KZb5EY0Q9GNqhivOyJPGX5STkGnF3isq)

Also to generate better prompts take some sample prompts from the [paper](https://aivideo.hunyuan.tencent.com/), [project page](https://aivideo.hunyuan.tencent.com/) and use a large language model to ask to devise a similar prompt for the idea you have in mind. 

To do list

Add support for parallel
Improve the prompt
