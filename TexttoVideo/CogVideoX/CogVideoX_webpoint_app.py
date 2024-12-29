import base64
import json
import os
import time
from typing import List
import requests
import streamlit as st
import torch
from uuid import uuid4
from PIL import Image

def save_metadata(
    prompt: str,
    converted_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
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
        "num_videos_per_prompt": num_videos_per_prompt,
        "manual_seed": manual_seed,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    """
    Main function to run the Streamlit web application.
    """
    st.set_page_config(page_title="CogVideoX-Demo", page_icon="🎥", layout="wide")
    st.write("# CogVideoX 🎥")

    tab1, tab2, = st.tabs(["Text to Video", "Image to Video"])

    with st.sidebar:
        st.info("It will take some time to generate a video (~600 seconds per videos in 50 steps).", icon="ℹ️")
        num_inference_steps: int = st.number_input("Inference Steps", min_value=1, max_value=100, value=50)
        guidance_scale: float = st.number_input("Guidance Scale", min_value=0.0, max_value=20.0, value=6.0)
        num_videos_per_prompt: int = st.number_input("Videos per Prompt", min_value=1, max_value=10, value=1)
        manual_seed: int = st.number_input("Manual Seed", min_value=1, max_value=9999, value=42)
        share_links_container = st.empty()

    with tab1:
        prompt: str = st.chat_input("Prompt")

        if prompt:
            # Not Necessary, Suggestions
            with st.spinner("Refining prompts..."):
                # converted_prompt = convert_prompt(prompt=prompt, retry_times=1)
                converted_prompt = prompt
                if converted_prompt is None:
                    st.error("Failed to Refining the prompt, Using origin one.")

            if converted_prompt is not None:
                st.info(f"**Origin prompt:**  \n{prompt}  \n  \n**Convert prompt:**  \n{converted_prompt}")
            else:
                st.info(f"**Origin prompt:**  \n{prompt}" )

            torch.cuda.empty_cache()

            with st.spinner("Generating Video..."):
                start_time = time.time()
                video_paths = []
                current_uuid = uuid4()
                print(current_uuid)
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"./output/{current_uuid}"
                os.makedirs(output_dir, exist_ok=True)

                metadata_path = os.path.join(output_dir, "config.json")
                save_metadata(
                    prompt, converted_prompt, num_inference_steps, guidance_scale, num_videos_per_prompt, manual_seed, metadata_path
                )

                for i in range(num_videos_per_prompt):
                    video_path = os.path.join(output_dir, f"output_{i + 1}.mp4")
                    
                    params = {
                        'prompt': converted_prompt,
                        'num_inference_steps': num_inference_steps,
                        'guidance_scale': guidance_scale,
                        'num_videos_per_prompt': num_videos_per_prompt,
                        'manual_seed': manual_seed,
                        'uuid_sent': current_uuid,
                    }
                    # Define the headers
                    headers = {
                        'accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
                    
                    url = st.secrets["Model_url"]["url"]
                    response = requests.get(url, headers=headers, params=params)

                    if response.status_code == 200:
                        
                        output_vid_base64 = response.content
                        video_data = base64.b64decode(output_vid_base64)
    
                        print(f"Saving it to {video_path}")
                        with open(video_path, "wb") as output_file:
                            output_file.write(video_data)
                    else:
                        print(f'Error: {response.status_code}')
                        print(response.text)
                    
                    video_paths.append(video_path)
                    with open(video_path, "rb") as video_file:
                        video_bytes: bytes = video_file.read()
                        st.video(video_bytes, autoplay=True, loop=True, format="video/mp4")
                    torch.cuda.empty_cache()

                used_time: float = time.time() - start_time
                st.success(f"Videos generated in {used_time:.2f} seconds.")

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

    with tab2:
        

        # Upload image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        # If an image is uploaded, display it
        if uploaded_file is not None:
            # Open the image using PIL
            image = Image.open(uploaded_file)
            
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)


        prompt2: str = st.text_input("Prompt")
if __name__ == "__main__":
    main()