# Streamlit Modal
Repo for applications created with streamlit frontend and modal as the backend

# Text to Image Application

## Flux
https://github.com/user-attachments/assets/8244d609-8c74-436f-b3e7-8c6e3d29e07e

The frontend has been created using streamlit that has a simple authenticator. The user can enter a prompt and change the number of generation steps, guidance scale and sequence length. It also allows the possibility of entering a manual seed number for image consistency, LORA path and file name for particualr style. The UI makes an API request to a serverless gpu infrastructure (Modal) that host the open source text to image model - Flux. The Modal server takes in the information, generates an image and sends it back as JSON response and is displayed to user. More details about the project can be found [here](https://medium.com/@avishekpaul31/how-to-create-web-applications-with-gpu-serverless-infrastructure-35eff89c74ed) and [here](https://github.com/AvisP/Streamlit_modal/tree/main/TexttoImage/Flux)

# Text to Video Application 

## CogVideoX 1.5 
https://github.com/user-attachments/assets/6c5d5fa8-b4f7-42fc-ac69-dcd778721498

A frontend has been developed with streamlit where the user can enter a prompt describing the scene they want to describe in a sequence. It has the option of entering number of inference steps, guidance scale, manual seed. There is an API created using modal container that takes in these values and generates a video. The video is saved on the modal container and also sent back to the streamlit application as base64 encoded content. An open source model CogVideoX 1.5B is used to generate the video. The application can further be enhanced by adding an authentication component.

### To Do
- [ ] Add Websockets and display video generation progress on the streamlit application
