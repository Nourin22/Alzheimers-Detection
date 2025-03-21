import streamlit as st
from PIL import Image
import base64
import numpy as np
import cv2
from keras.models import load_model

# Load the model
model = load_model("Model/model.keras")
classes = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}

def predict(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 128, 128, 1)
    pred = model.predict(image)
    return pred

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def style_app():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #b3b6b7 !important;
            height: 80vh !important;
            width: 200px !important;
            overflow: hidden !important;
            border-radius: 10px;
            position: absolute !important;
            top: 100px !important;
        }
        html, body, [class*="st-"] {
            font-family: 'Times New Roman', serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Times New Roman', serif !important;
        }
        [data-testid="stSidebar"] * {
            color: black !important;
        }
        .main .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


style_app()

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = "file_uploader_" + str(np.random.randint(1000))

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Alzheimer's Prediction"])

if page == "Home":
    set_background("bg.jpg")

    st.markdown(
        """
        <style>
        .main, .block-container {
            color: white !important;
            text-align: justify !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    """
    <h1 style="text-align: center;">
        🧠 Alzheimer’s Disease: Overview & <br> 
        <span style="letter-spacing: 1.5px;">Early Detection</span>
    </h1>
    """,
    unsafe_allow_html=True
)

    st.write("""
    **Alzheimer’s disease** is a progressive neurodegenerative disorder and the leading cause of dementia. It affects memory, cognitive function, and behavior, gradually impairing daily life. While its exact cause remains unclear, factors such as genetics, aging, and vascular health contribute to its development.
    
    ### Key Symptoms:
    - **Memory Impairment** – Difficulty recalling recent events or conversations  
    - **Cognitive Decline** – Struggles with problem-solving, decision-making, and language  
    - **Disorientation** – Confusion about time, place, or familiar surroundings  
    - **Behavioral Changes** – Mood swings, depression, or withdrawal from social activities  
    
    ### 🧬 Deep Learning-Based Early Detection
    Our advanced **deep learning model** leverages Convolutional Neural Networks (CNNs) to analyze MRI brain images for early **Alzheimer’s detection**. By processing and classifying medical imaging data, our AI-driven system enhances diagnostic accuracy, aiding clinicians in early intervention and patient care.
    
    **Navigate to the Alzheimer's Prediction page to upload and analyze MRI scan images.**
    """)

elif page == "Alzheimer's Prediction":
    st.title("Alzheimer's Detection")
    st.write("Upload an MRI brain scan image for analysis.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"], key=st.session_state.file_uploader_key)

    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.session_state.uploaded_image = np.array(st.session_state.uploaded_image)
        st.session_state.uploaded_image = cv2.resize(st.session_state.uploaded_image, (128, 128))
    
    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Predict"):
                result = predict(st.session_state.uploaded_image)
                result = np.argmax(result)
                st.session_state.prediction = classes[result]
                st.write(f"### Prediction: {st.session_state.prediction}")

        with col2:
            if st.button("Close"):
                st.session_state.uploaded_image = None
                st.session_state.prediction = None
                st.session_state.file_uploader_key = "file_uploader_" + str(np.random.randint(1000))
                st.rerun()
