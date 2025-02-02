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
import json
from huggingface_hub import repo_exists, file_exists
import re
import threading
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

def display_messages():
    for msg in st.session_state["chat_history"][1:]:
        print(msg["role"])
        if msg["role"] == "user":
            selected_avatar = "👨‍💻"
        else:
            selected_avatar = "🤖"
        st.chat_message(msg["role"], avatar=selected_avatar).write(msg["content"])

# Define a global variable to hold the thread and stop flag
current_thread = None
stop_flag = threading.Event()

url_unsloth = st.secrets["url_unsloth"]
if st.session_state['authentication_status']:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # st.session_state.chat_history.append({ "role": "user", "content": "You are a friendly chatbot who answers user questions"})
        # st.session_state.chat_history.append({"role": "assistant", "content": "Hello! I'm a friendly chatbot here to help answer any questions you may have. What's on your mind today?"})
    
    st.title('Demo Application for Deep Seek R1 🐋')

    with st.sidebar:
        st.sidebar.title("Settings")
        temperature = st.sidebar.slider(
                "Temperature",
                min_value=0.01,  # Minimum value
                max_value=4.0,  # Maximum value
                value=0.01, # Default value
                step=0.005  # Step size
            )
        
        predict_next_tokens = st.sidebar.slider(
                "predict_next_tokens",
                min_value=8,  # Minimum value
                max_value=8192,  # Maximum value
                value=4096, # Default value
                step=16  # Step size
            )
        
        context_length = st.sidebar.slider(
                "context_length",
                min_value=16,  # Minimum value
                max_value=8192,  # Maximum value
                value=4092, # Default value
                step=8  # Step size
            )
        
        top_p = st.sidebar.slider(
                "Top_p",
                min_value=0.01,  # Minimum value
                max_value=1.0,  # Maximum value
                value=0.75, # Default value
                step=0.005  # Step size
            )
        
        top_k = st.sidebar.slider(
                "Top_k",
                min_value=100,  # Minimum value
                max_value=1,  # Maximum value
                value=40, # Default value
                step=1  # Step size
            )
        manual_seed = st.number_input("Manual Seed", min_value=1, max_value=9999, value=1983)
        model_entrypoint_file = st.text_input("Path to Qunatized DS R1 model", help = "Path to stored Model weights, can be noted by doing a modal shell into the modal volume eg. DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf")
        debug_flag = st.checkbox('Debug prompt history')

    with st.container():
        if st.button("Reset Chat"):
                st.session_state.chat_history = []
                # st.session_state.chat_history.append({ "role": "user", "content": "You are a friendly chatbot who answers user questions"})
                # st.session_state.chat_history.append({"role": "assistant", "content": "Hello! I'm a friendly chatbot here to help answer any questions you may have. What's on your mind today?"})
                
        user_query = st.chat_input("Enter your prompt", max_chars=3000)
        display_messages()
        
        if user_query:
            st.chat_message("user", avatar="👨‍💻").write(user_query)
                
            st.session_state.chat_history.append({ "role": "user", "content": user_query})

            if debug_flag:
                    st.write(st.session_state.chat_history)

            params = {
                    'model_entrypoint_file': model_entrypoint_file,
                    'prompt': user_query,
                    'n_predict': predict_next_tokens, 
                    'top_p': top_p,
                    'top_k': top_k,
                    'temperature': temperature,
                    'do_sample': True,
                    'seed': manual_seed,
                    'store_output':True,
                    'ctx-size': context_length
                }

            # Define the headers
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }

            # Perform the GET request
            with st.spinner():
                response = requests.get(url_unsloth, headers=headers, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                if response.status_code == 200:
                    print("Recieved response")
                    print(response.text)
                    # Convert list to a single string
                    joined_string = ''.join(response.text)

                    # Remove special characters using a regex to find and remove ANSI escape sequences
                    cleaned_string = re.sub(r'\x1b\[[0-9;]*m', '', joined_string)
                    # cleaned_string.replace('\\u001b[33m', '').replace('\\u001b[0m', '')
                    response_dict = cleaned_string.replace('\\u001b[33m', '').replace('\\u001b[0m', '').replace('</think>', '').replace('[end of text]', '').replace('<think>', '')
                    response_dict.replace(user_query, '')
                    response_dict2 = response_dict[1:-1].replace('\\n\\n\\n', '')

                    print(response_dict)
                
                st.chat_message("assistant", avatar="🤖").write(response_dict2)
                st.session_state.chat_history.append({ "role": "assistant", "content": response_dict2})
            else:
                print(f'Error: {response.status_code}')
                print(response.text)
                st.write(f'Error: {response.status_code}')
                st.write(response.text)
            
