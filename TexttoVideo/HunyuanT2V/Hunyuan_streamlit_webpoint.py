import base64
import json
import os
import time
import threading
import requests
import streamlit as st
import queue
from uuid import uuid4
from websocket import WebSocketApp
import string

# WebSocket server URI
start_url = st.secrets["Model_url"]["start_url"]
result_url = st.secrets["Model_url"]["result_url"] 
cancel_url = st.secrets["Model_url"]["cancel_url"]   
WS_URI = url = st.secrets["Model_url"]["WS_URI"]

# Set your reconnect interval and backoff time if needed
RECONNECT_INTERVAL = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 10  # maximum number of reconnection attempts

# Global variable to hold the WebSocket object
wsapp = None

# Global variable to store WebSocket message for progress display
progress_message = ""

def save_metadata(
    prompt: str,
    converted_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    embedded_guidance_scale: float,
    flow_shift: float,
    num_videos_per_prompt: int,
    manual_seed: int,
    path: str,
) -> None:
    """
    Save metadata to a JSON file.

    Args:
    - prompt (str): Original prompt.
    - converted_prompt (str): Converted prompt.
    - num_inference_steps (int): Number of inference steps.
    - guidance_scale (float): Guidance scale.
    - num_videos_per_prompt (int): Number of videos per prompt.
    - path (str): Path to save the metadata.
    """

    metadata = {
        "prompt": prompt,
        "converted_prompt": converted_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "embedded_guidance_scale": embedded_guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
        "manual_seed": manual_seed,
        "flow_shift": flow_shift
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

def slugify(prompt, i):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:100]  # since filenames can't be longer than 255 characters
    mp4_name = str(int(time.time())) + "_" + prompt + "_" + str(i+1) +".mp4"
    return mp4_name
    
# Reconnection logic
def connect_websocket():
    global wsapp  # Access the global wsapp variable to store the WebSocketApp
    reconnect_attempts = 0
    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            # Create a new WebSocketApp instance
            wsapp = WebSocketApp(WS_URI, 
                                 on_message=on_message, 
                                 on_error=on_error, 
                                 on_close=on_close, 
                                 on_open=on_open)
            # Start the WebSocketApp in a separate thread
            wsapp.run_forever()
            break  # Break if run_forever exits normally (connected and closed cleanly)
        except Exception as e:
            print(f"Error while trying to connect: {e}")
            reconnect_attempts += 1
            print(f"Reconnecting in {RECONNECT_INTERVAL} seconds... (Attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})")
            time.sleep(RECONNECT_INTERVAL)
    if reconnect_attempts == MAX_RECONNECT_ATTEMPTS:
        print("Max reconnection attempts reached. Giving up.")

# WebSocket functions (callbacks)
def on_message(ws, message):
    global progress_message
    if (message == "ping"):
        print("Ping received. Sending pong...")
        ws.send_text("pong")
    else:
    # Update the global variable with the message, which will be displayed on Streamlit
        print(message)
        progress_message = message

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)
    if close_status_code != 1000:  # Non-clean close (code 1000 is a normal closure)
        print("Reconnecting...")
        connect_websocket()

def on_error(ws, error):
    print("WebSocket error:", error)

def on_open(wsapp):
    print("WebSocket connection established")

# Start WebSocket in a separate thread
def start_websocket():
    websocket_thread = threading.Thread(target=connect_websocket)
    websocket_thread.daemon = True  # Ensures the thread exits when the main program exits
    websocket_thread.start()

# Function to cleanly close the WebSocket connection
def close_websocket():
    global wsapp
    if wsapp:
        print("Closing WebSocket connection...")
        wsapp.close()


def make_request(url, headers, params):
    """
    Makes a GET request in a separate thread.

    Args:
        url: The URL to make the request to.
        headers: The headers to include in the request.
        params: The parameters to include in the request.

    Returns:
        The response object from the request.
    """
    try:
        response = requests.get(url, headers=headers, params=params)
        result_queue.put(response) 
        return response
    except Exception as e:
        print(f"Error making request: {e}")
        result_queue.put(None) 
        return None
    
# Create a queue to store the response
result_queue = queue.Queue()

import re

def extract_progress_parts(text):
  """
  Extracts the "x/y" progress parts from the given text, 
  separating "Loading" and "General" progress.

  Args:
    text: The input text containing progress information.

  Returns:
    A tuple containing two lists:
      - loading_progress: A list of "x/y" strings for "Loading" progress.
      - general_progress: A list of "x/y" strings for "General" progress.
  """
  loading_pattern = r"Loading checkpoint shards:.*?(\d+/\d+)"
  general_pattern = r"^(?!Loading checkpoint shards:).*?(\d+/\d+)" 

  loading_progress = re.findall(loading_pattern, text)
  general_progress = re.findall(general_pattern, text, flags=re.MULTILINE) 

  return loading_progress, general_progress


def main() -> None:
    """
    Main function to run the Streamlit web application.
    """
    st.set_page_config(page_title="HunYuan Text to Video", page_icon="🎥", layout="wide")
    st.write("# HunYuan Text to Video 🎥")

    with st.sidebar:
        st.info("It will take some time to generate a video (~60 seconds per inference step of the video so 30 inference steps would take 1800).", icon="ℹ️")
        num_inference_steps: int = st.number_input("Inference Steps", min_value=1, max_value=100, value=30)
        guidance_scale: float = st.number_input("Guidance Scale", min_value=0.0, max_value=20.0, value=1.0)
        embedded_guidance_scale: int = st.number_input("Embedded Guidance Scale", min_value=1.0, max_value=10.0, value=6.0)
        flow_shift: int = st.number_input("Flow Shift", min_value=1, max_value=25, value=7)
        manual_seed: int = st.number_input("Manual Seed", min_value=1, max_value=9999, value=42)
        num_videos_per_prompt : int = st.number_input("Number of Videos per prompt", min_value=1, max_value=4, value=1)
        share_links_container = st.empty()

    prompt: str = st.chat_input("Prompt")

    if prompt:
        # Not Necessary, Suggestions
        with st.spinner("Refining prompts..."):
            # converted_prompt = convert_prompt(prompt=prompt, retry_times=1)
            converted_prompt = prompt
            if converted_prompt is None:
                st.error("Failed to refine the prompt, Using origin one.")

        if converted_prompt is not None:
            st.info(f"**Origin prompt:**  \n{prompt}  \n  \n**Convert prompt:**  \n{converted_prompt}")
        else:
            st.info(f"**Origin prompt:**  \n{prompt}" )

        # Start the video generation process with WebSocket communication
        start_time = time.time()
        video_paths = []
        current_uuid = uuid4()
        print(current_uuid)

        output_dir = f"./output/"
        os.makedirs(output_dir, exist_ok=True)

        metadata_path = os.path.join(output_dir, "config.json")
        save_metadata(
            prompt, converted_prompt, num_inference_steps, guidance_scale, embedded_guidance_scale, flow_shift, num_videos_per_prompt, manual_seed, metadata_path
        )

        # Create a placeholder for progress updates in Streamlit
        placeholder = st.empty()
        # cancel_button_placeholder = st.empty()

        # Start the WebSocket connection and handle video processing concurrently
        start_websocket()   

        for i in range(num_videos_per_prompt):
            mp4_name = slugify(prompt, i)
            video_path = os.path.join(output_dir, mp4_name)
            print(video_path)
            params = {
                'prompt': converted_prompt,
                'infer_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'embedded_guidance_scale': embedded_guidance_scale,
                'flow_shift': flow_shift,
                'flow_reverse': True,
                'num_videos_per_prompt': 1,
                'manual_seed': manual_seed,
                'uuid_sent': current_uuid,
            }
            # Define the headers
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }

            # Perform the GET request
            response_function_call_id = requests.get(start_url, headers=headers, params=params)
            if response_function_call_id.status_code == 200:
                function_call_id = json.loads(response_function_call_id.content.decode('utf-8'))['function_call_id']
                print("Function_call_id :", function_call_id)
            else:
                st.error("Process spawn request error", str(response_function_call_id.status_code))
                print("Process spawn request error", str(response_function_call_id.status_code))
                function_call_id = None
            
            if function_call_id:
                video_process_thread = threading.Thread(target=make_request, args=(result_url, headers, {"function_call_id": function_call_id}))
                video_process_thread.start()
                cancel_action = st.button("Cancel")

                # video_process_thread.join()  # Wait for the thread to complete
                while video_process_thread.is_alive():
                    loading_checkpoint_progress, general_progress = extract_progress_parts(progress_message)
                    if cancel_action:
                        cancel_response = requests.get(cancel_url, headers=headers, params={"function_call_id": function_call_id})
                        cancel_status = json.loads(cancel_response.content.decode('utf-8'))['cancelled']
                        print("Cancel Status ", cancel_status)
                        if cancel_status:
                            st.warning("Task cancelled")
                            break
                    if general_progress:
                        # print(f"{progress_type} Progress: {progress}")
                        last_value_general = general_progress[-1]
                        placeholder.progress(int(last_value_general.split("/")[0])/int(last_value_general.split("/")[1]) , text='Generating Video')
                    elif loading_checkpoint_progress:
                        # print(f"{progress_type} Progress: {progress}")
                            last_value_loading_checkpoint = loading_checkpoint_progress[-1]
                            placeholder.progress(int(last_value_loading_checkpoint.split("/")[0])/int(last_value_loading_checkpoint.split("/")[1]) , text='Loading Checkpoints')
                    else:
                        placeholder.text("Initializing")
                    time.sleep(5)
                else:
                    # Access the result
                    response = result_queue.get()  # Get the result from the queue
                    
                    if response.status_code == 200:
                        placeholder.text("Video generation completed!")
                        output_vid_base64 = response.content
                        video_data = base64.b64decode(output_vid_base64)

                        with open(video_path, "wb") as output_file:
                            output_file.write(video_data)
                        print(f"Saving it to {video_path}")

                        video_paths.append(video_path)
                        with open(video_path, "rb") as video_file:
                            video_bytes: bytes = video_file.read()
                            st.video(video_bytes, autoplay=True, loop=True, format="video/mp4")

                        used_time = time.time() - start_time
                        st.success(f"Videos generated in {used_time:.2f} seconds.")
                    else:
                        error_message = f"Error: {response.status_code}, {response.text}"
                        print(error_message)
                        st.error(error_message)

        # Close the WebSocket connection when done
        close_websocket()

        # Create download links in the sidebar
        with share_links_container:
            st.sidebar.write("### Download Links:")
            for video_path in video_paths:
                video_name = os.path.basename(video_path)
                with open(video_path, "rb") as f:
                    video_bytes: bytes = f.read()
                b64_video = base64.b64encode(video_bytes).decode()
                href = f'<a href="data:video/mp4;base64,{b64_video}" download="{video_name}">Download {video_name}</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()