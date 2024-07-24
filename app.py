import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace

# Function to detect emotion
def detect_emotion(image):
    try:
        # Use DeepFace to analyze the image
        analysis = DeepFace.analyze(image, actions=['emotion'])
        # Extract the dominant emotion
        if isinstance(analysis, dict):
            emotion = analysis['dominant_emotion']
        elif isinstance(analysis, list):
            emotion = analysis[0]['dominant_emotion']
        return emotion
    except Exception as e:
        return None

# Streamlit UI
st.title("Emotion Detection from Image")
st.write("Upload an image and the application will try to detect the emotion of the person in the image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert image to numpy array
    img_array = np.array(image)

    # Detect emotion
    emotion = detect_emotion(img_array)
    if emotion:
        st.write(f"Detected Emotion: {emotion.capitalize()}")
    else:
        st.write("No emotion detected or the image is not of a person.")
