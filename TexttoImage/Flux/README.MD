This repo contains code for creating an API on serverless GPU [Modal](https://modal.com/docs) for the popular text to image generation model Flux. 

The modal app can be created after creating an account with Modal and then installing it using `pip install modal`. For starting the modal app after authenticating and an api can be deployed by running `modal serve Modal_Flux_app.py`. To test the modal file locally it can be run using `modal run Modal_Flux_app.py` and prompts and other parameters can be changed on lines 207 - 221 of `Modal_Flux_app.py`. To test the webpoint functionalities it can be done by commenting lines 207-221 and uncommenting 223-259. Then test by `modal run Modal_Flux_app.py` which will make a post request to the endpoint using the requests library. The API link needs to be substituted in the secrets.toml file for streamlit to use to make calls.

The frontend is developed using `streamlit` which can be installed by doing `pip install streamlit`. An authentication component is also needed which can be installed by doing `pip install streamlit-authenticator` The application can be started by doing `streamlit run streamlit_modal_app.py`. Also on the client side a data folder needs to be created that will contain the authentication credentials in json format or added into the secrets.toml file. If a json file is created in different folder then it needs to be updated on the lines 20 to 22 of `streamlit_modal.py`. For local or other development using streamlit, it should be put into `.streamlit/secrets.toml` as shown in the example. There is option to enter these secrets if put on the streamlit community cloud.

The frontend after successful authentication gives access to the apploication which allows the user to change a number of different parameters like `Inference Steps`, `Guidance Scale`, 'Maximum Sequence Length` and option to set `manual seed` and on the center option to enter a prompt. It typically takes about 90 seconds to generate an image for 20 steps and generation time increases in proprotion to increase of number of steps. A bit longer on the first run as the model weights need to be loaded.

Some examples of image generated using this application

![IMG1](img&vid/SampleImage.png)

![IMG2](img&vid/SampleImage3.png)

![IMG3](img&vid/b5861826defc457dcfd92996971d2e69e84499b1163b1b6ede8acd97.jpg)
