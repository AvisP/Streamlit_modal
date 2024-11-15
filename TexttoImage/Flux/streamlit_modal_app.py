import requests
import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)
print(st.__version__)
from PIL import Image
import base64
from io import BytesIO
import json
from huggingface_hub import repo_exists, file_exists

# # Loading config file
# with open('./data/config.yaml', 'r', encoding='utf-8') as file:
#     config = yaml.load(file, Loader=SafeLoader)

config = {
    'credentials': {
        'usernames': {
            username: {
                'email': user_data['email'],
                'failed_login_attempts': user_data['failed_login_attempts'],
                'logged_in': user_data['logged_in'],
                'name': user_data['name'],
                'password': user_data['password']
            }
            for username, user_data in st.secrets['usernames'].items()
        }
    },
    'cookie': {
        'expiry_days': st.secrets['cookie']['expiry_days'],
        'key': st.secrets['cookie']['key'],
        'name': st.secrets['cookie']['name']
    },
    'pre-authorized': {
        'emails': st.secrets['pre-authorized']['emails']
    }
}

# Hashing all plain text passwords once
# Hasher.hash_passwords(config['credentials'])

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    # auto_hash=True,
)

# Creating a login widget
try:
    authenticator.login(location="sidebar")

except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    authenticator.logout(location="sidebar")
    st.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# Saving config file
with open('../config.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(config, file, default_flow_style=False)

# Creating a password reset widget
if st.session_state['authentication_status']:
    try:
        st.title('Demo Application for Flux text to image on Modal')

        with st.sidebar:
            st.sidebar.title("Settings")
            num_inference_steps = st.sidebar.slider(
                    "Inference Steps",
                    min_value=1,  # Minimum value
                    max_value=100,  # Maximum value
                    value=20, # Default value
                    step=1  # Step size
                )
            
            guidance_scale = st.sidebar.slider(
                    "Guidance scale",
                    min_value=0.1,  # Minimum value
                    max_value=13.0,  # Maximum value
                    value=7.0, # Default value
                    step=0.1  # Step size
                )
            
            max_sequence_length = st.sidebar.slider(
                    "Maximum Sequence Length",
                    min_value=0,  # Minimum value
                    max_value=1024,  # Maximum value
                    value=256, # Default value
                    step=16  # Step size
                )
            
            enable_manual_seed = st.checkbox("Enable Manual Seed")
            use_lora = st.checkbox("Enable Lora")

            if enable_manual_seed:
                seed_input = st.text_input("Enable Manual Seed")
            else:
                seed_input = 0

            if use_lora:
                lora_path = st.text_input("Lora path name", help="Enter the huggingface lora path")
                lora_weight = st.text_input("Lora weight name", help = "Enter the safetensor name in the huggingface path")
            else:
                lora_path = None
                lora_weight = None
        

        with st.container():
            prompt = st.text_input("Enter your prompt", value="", max_chars=500)
            process_button = st.button("Process", type="primary")

        if process_button:
            if use_lora:
                if len(lora_path)>0:
                    if not repo_exists(lora_path):
                        st.warning('Lora repo may not exist, check path', icon="⚠️")
                    else:
                        if len(lora_weight)>0:
                            if "safetensors" not in  lora_weight:
                                lora_weight += lora_weight + ".safetensors"
                            if not file_exists(lora_path, lora_weight):
                                st.warning('Lora file may not exist, check file name', icon="⚠️")
                        else:
                            st.warning('Lora file is empty, check file name', icon="⚠️")

                else:
                    st.warning('Lora repo is empty, check path', icon="⚠️")
                

            with st.spinner('Processing Request (takes about 30 secs)...'):
                url = st.secrets["url"] 
                params = {
                    'prompt': prompt,
                    'n_steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'max_sequence_length': max_sequence_length,
                    'manual_seed': seed_input,
                    'lora_path': lora_path,
                    'lora_weight': lora_weight,
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
                    data = response.json()  # Parse the JSON response

                    encoded_image = data['image']
                    request_id = data['request_id']


                    image_data = base64.b64decode(encoded_image)
                    image = Image.open(BytesIO(image_data))

                    st.write("Request ID ", json.loads(request_id)['uuid'])
                    output_path ="output1.png"
                    print(f"Saving it to {output_path}")
                    image.save(output_path)
                    st.image(output_path, caption=prompt)
                else:
                    print(f'Error: {response.status_code}')
                    print(response.text)
    except (CredentialsError, ResetError) as e:
        st.error(e)