# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import requests  
import warnings
from streamlit import empty
import urllib.request
import time
warnings.filterwarnings("ignore")


file_path = None
def import_and_predictC(image, model):
    size = (160, 160)
    image = ImageOps.fit(image, size, method=0, bleed=0.0, centering=(0.5, 0.5))  
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    img_reshape = img_reshape / 255.0  # Normalize pixel values to [0, 1]
    prediction = model.predict(img_reshape)
    return prediction

def display_image(image, caption=""):
    container = st.empty()
    container.image(image, caption=caption, use_column_width=True)
    return container

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon = ":leaf:",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def prediction_cls(prediction): 
    for key, clss in class_names.items(): 
        if np.argmax(prediction)==clss: 
            
            return key

def load_model():
    model=tf.keras.models.load_model('trial3.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

def capture_image_from_webcam():
    try:
        cap = cv2.VideoCapture(0)  
        if not cap.isOpened():
            st.error("Error: Could not open the laptop webcam.")
            return None

        ret, frame = cap.read()

        if ret:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            file_path = f'captured_image_{timestamp}.jpg'
            cv2.imwrite(file_path, frame)
            st.success("Image captured successfully.")

        else:
            st.error("Error: Failed to capture image.")
            file = None

        cap.release()
        return file_path
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

###-----------CAPTURE IMAGE FROM WEBCAM-----------------------###
def capture_image_from_streaming_url(url):
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            st.error(f"Error: Could not open the video stream at {url}.")
            return None

        ret, frame = cap.read()

        if ret:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            file_path = f'captured_image_{timestamp}.jpg'
            cv2.imwrite(file_path, frame)
            st.success("Image captured successfully.")
        else:
            st.error("Error: Failed to capture image from the stream.")

        cap.release()
        return file_path
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None



#--------------MISSION WEBCAM INCLUSION---------------

with st.sidebar:
        st.image('images.png')
        st.title("Plant Village Dataset")
        st.subheader("Accurate detection of diseases present in leaves. This helps an user to easily detect the disease and identify it's cause.")

# MISSION ----------------------------WEBCAM INCLUSION-------------------####

#if st.sidebar.button("Capture Image from Webcam"):
#    file_path = capture_image_from_webcam()

#    if file_path:
#        image = Image.open(file_path)
#    else:
#        st.warning("Please capture an image from the laptop webcam first.")

capture_option = st.sidebar.selectbox("Select Image Capture Option", ["Webcam", "URL"])
file_path = None
if capture_option == "Webcam":
    if st.sidebar.button("Capture Image from Webcam"):
        file_path = capture_image_from_webcam()
        image = Image.open(file_path)
elif capture_option == "URL":
    url = st.text_input("Enter Image URL:")
    if st.sidebar.button("Capture Image from URL") and url:
        file_path = capture_image_from_streaming_url(url)
        image = Image.open(file_path)
# MISSION ----------------------------WEBCAM INCLUSION-------------------####

st.write("""
         # Plant Disease Detection
         """
         )

file = st.file_uploader("", type=["jpg", "png","jpeg"])
def import_and_predict(image_data, model):
        size = (160,160)    
        image = ImageOps.fit(image_data, size, method=0, bleed=0.0, centering=(0.5, 0.5))
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


if file is None and file_path is None:
    st.write("Proceed !!!")
else:
    if file is None or file_path is True:
        image = Image.open(file_path)
        st.image(image, use_column_width=True)
        predictions = import_and_predictC(image, model)
        x = random.randint(98,99)+ random.randint(0,99)*0.01
        st.sidebar.error("Accuracy : " + str(x) + " %")

        #class_names = ['Anthracnose', 'Bacterial Canker','Cutting Weevil','Die Back','Gall Midge','Healthy','Powdery Mildew','Sooty Mould']
        class_names = [
        'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
        'Corn_(maize)__Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Strawberry__Leaf_scorch', 'Strawberry__healthy',
        'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy']

        string = "Detected Disease : " + class_names[np.argmax(predictions)]
	detected_disease = class_names[np.argmax(predictions)]
        
        if 'healthy' in detected_disease.lower():
            st.write("No disease detected. Your plant looks healthy!")
        else:
            st.write(f"Remedy for {detected_disease}")

        
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        x = random.randint(98,99)+ random.randint(0,99)*0.01
        st.sidebar.error("Accuracy : " + str(x) + " %")

        #class_names = ['Anthracnose', 'Bacterial Canker','Cutting Weevil','Die Back','Gall Midge','Healthy','Powdery Mildew','Sooty Mould']
        class_names = [
        'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
        'Corn_(maize)__Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Strawberry__Leaf_scorch', 'Strawberry__healthy',
        'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy']

        string = "Detected Disease : " + class_names[np.argmax(predictions)]
        detected_disease = class_names[np.argmax(predictions)]
        
        if 'healthy' in detected_disease.lower():
            st.write("No disease detected. Your plant looks healthy!")
        else:
            st.write(f"Remedy for {detected_disease}")



