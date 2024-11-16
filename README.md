# Streamlit Modal
Repo for applications created with streamlit frontend and modal as the backend

## Text to Image Application
https://github.com/user-attachments/assets/8244d609-8c74-436f-b3e7-8c6e3d29e07e

The frontend has been created using streamlit that has a simple authenticator. The user can enter a prompt and change the number of generation steps, guidance scale and sequence length. It also allows the possibility of entering a manual seed number for image consistency, LORA path and file name for particualr style. The UI makes an API request to a serverless gpu infrastructure (Modal) that host the open source text to image model - Flux. The Modal server takes in the information, generates an image and sends it back as JSON response and is displayed to user. More details about the project can be found [here](https://medium.com/@avishekpaul31/how-to-create-web-applications-with-gpu-serverless-infrastructure-35eff89c74ed)
