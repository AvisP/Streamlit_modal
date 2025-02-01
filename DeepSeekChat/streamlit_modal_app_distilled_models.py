import requests
import yaml
import streamlit as st
from yaml.loader import SafeLoader
import time
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)
import subprocess
import json
from huggingface_hub import repo_exists, file_exists
import time
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
        # print(msg["role"])
        if msg["role"] == "user":
            selected_avatar = "👨‍💻"
        else:
            selected_avatar = "🤖"
        st.chat_message(msg["role"], avatar=selected_avatar).write(msg["content"])

# Define a global variable to hold the thread and stop flag
current_thread = None
seleted_model_option = None
stop_flag = threading.Event()

def run_modal(option):
    """Run the modal command based on the option."""
    while not stop_flag.is_set():
        try:
            if option == "Llama-8B":
                subprocess.run(['modal', 'serve', 'Deepseekr1_modal_lmdeploy_Llama_8B.py'])
            elif option == "Qwen-14B":
                subprocess.run(['modal', 'serve', 'Deepseekr1_modal_lmdeploy_Qwen_14B.py'])
            elif option == "Qwen-32B":
                subprocess.run(['modal', 'serve', 'Deepseekr1_modal_lmdeploy_Qwen_32B.py'])
            elif option == "Llama-70B":
                subprocess.run(['modal', 'serve', 'Deepseekr1_modal_lmdeploy_Llama_70B.py'])
            
            break  # Exit the loop when the subprocess completes
        except Exception as e:
            st.error(f"Error occurred: {e}")
            break  # Exit the loop in case of an error
        time.sleep(1)  # Sleep to avoid hogging CPU

def start_new_thread(option, placeholder):
    """Start a new thread for the selected option."""
    global current_thread, stop_flag
    
    
    # Stop the existing thread if there is one
    if current_thread is not None:
        stop_flag.set()  # Signal the existing thread to stop
        current_thread.join()  # Wait for the existing thread to finish
        stop_flag.clear()  # Reset the stop flag
        # st.rerun()  # Rerun Streamlit to update UI after stopping the mode

    placeholder.info('Starting Model ' + option, icon="ℹ️")
    # Start a new thread with the updated option
    current_thread = threading.Thread(target=run_modal, args=(option,))
    current_thread.start()
    time.sleep(2)
    placeholder.info('Running Model ' + option, icon="ℹ️")

url = st.secrets["url"]
if st.session_state['authentication_status']:

    if "model_option" not in st.session_state:
        st.session_state.model_option = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # st.session_state.chat_history.append({ "role": "user", "content": "You are a friendly chatbot who answers user questions"})
        # st.session_state.chat_history.append({"role": "assistant", "content": "Hello! I'm a friendly chatbot here to help answer any questions you may have. What's on your mind today?"})
    
    st.title('Demo Application for Deep Seek 🐋')

    with st.sidebar:
        st.sidebar.title("Settings")
        temperature = st.sidebar.slider(
                "Temperature",
                min_value=0.01,  # Minimum value
                max_value=4.0,  # Maximum value
                value=0.6, # Default value
                step=0.005  # Step size
            )
        
        max_new_tokens = st.sidebar.slider(
                "max new tokens",
                min_value=8,  # Minimum value
                max_value=4096,  # Maximum value
                value=256, # Default value
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

        debug_flag = st.checkbox('Debug prompt history')

        selected_model_option = st.selectbox("Select model",
                    ("Llama-8B", "Qwen-14B", "Qwen-32B", "Llama-70B"),
                )
        placeholder = st.empty()
        # Start a new thread whenever the option changes
        if selected_model_option != st.session_state["model_option"]:
            st.session_state["model_option"] = selected_model_option
            start_new_thread(selected_model_option, placeholder)
        
            
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
                'prompt': json.dumps(st.session_state.chat_history),
                'max_new_tokens': max_new_tokens, 
                'top_p': top_p,
                'top_k': top_k,
                'temperature':temperature,
                'do_sample': True,
                'manual_seed': manual_seed,
            }
            # Define the headers
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }

            # Perform the GET request
            with st.spinner('Processing....'):
                response = requests.get(url, headers=headers, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                response_dict = json.loads(response.text)['text']
                if 'User' in response_dict:
                    selective_response_dict = response_dict.split('User')[0]   ### As endless chat happening
                    print(selective_response_dict)
                    selective_response_dict2 =  selective_response_dict.replace('</think>', '')[1:]
                else:
                    selective_response_dict2 = response_dict.replace('</think>', '')[1:]
                st.chat_message("assistant", avatar="🤖").write(selective_response_dict2)
                st.session_state.chat_history.append({ "role": "assistant", "content": selective_response_dict2})
            else:
                print(f'Error: {response.status_code}')
                print(response.text)
            
