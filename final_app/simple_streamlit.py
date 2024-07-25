import streamlit as st
import requests
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from gtts import gTTS
from pydub import AudioSegment
from textblob import TextBlob
import os
import nltk
from nltk.tokenize import sent_tokenize
import base64
from deepface import DeepFace
import json

def generate_download_link(content, filename, content_type):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{content_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Set page config
st.set_page_config(
    page_title="AI-Powered Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to save feedback to a JSON file
def save_feedback(feedback_list):
    with open('feedback.json', 'w') as f:
        json.dump(feedback_list, f)

# Download required NLTK data
nltk.download('punkt')

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Load Faster R-CNN model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights_path = 'model_epoch_5.pth'

def load_faster_rcnn_model(weights_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 91  # COCO dataset classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

faster_rcnn_model = load_faster_rcnn_model(weights_path, device)

# Load Detectron2 model
def load_detectron2_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor

detectron2_predictor = load_detectron2_model()

# Transform
transform = T.Compose([
    T.Resize((800, 800)),  # Resize to a fixed size
    T.ToTensor(),
])

def predict_faster_rcnn(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = faster_rcnn_model(image_tensor)
    output = outputs[0]
    labels = output['labels'].cpu().numpy().tolist()
    scores = output['scores'].cpu().numpy().tolist()
    detections = []
    for label, score in zip(labels, scores):
        if score > 0.5:  # Adjust the threshold as needed
            detection = {
                'class': COCO_CLASSES[label],
                'score': float(score)
            }
            detections.append(detection)
    return detections

def predict_detectron2(image):
    image_array = np.array(image)
    outputs = detectron2_predictor(image_array)
    labels = outputs["instances"].pred_classes.cpu().numpy().tolist()
    scores = outputs["instances"].scores.cpu().numpy().tolist()
    detections = []
    for label, score in zip(labels, scores):
        if score > 0.5:  # Adjust the threshold as needed
            detection = {
                'class': COCO_CLASSES[label + 1],  # Detectron2 labels start at 0
                'score': float(score)
            }
            detections.append(detection)
    return detections

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

# Function to convert text to audio and save it
def text_to_audio(text, language='en', filename='speech.mp3'):
    tts = gTTS(text=text, lang=language)
    tts.save(filename)
    return filename

# Function to add background music to an audio segment
def add_background_music_to_segment(speech_path, background_path, output_path="output_segment.mp3"):
    speech = AudioSegment.from_mp3(speech_path)
    background = AudioSegment.from_mp3(background_path)
    
    # Adjust the volume of the background music
    background = background - 20
    
    # Trim the background music to the length of the speech + 2 seconds
    background = background[:len(speech) + 2000]
    
    # Combine the speech and background music
    combined = background.overlay(speech)
    
    # Export the combined audio
    combined.export(output_path, format="mp3")
    return output_path

# Function to detect the mood of a text segment
def detect_mood(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.1:
        return "Happy"
    elif sentiment < -0.1:
        return "Sad"
    else:
        return "Calm"

# Function to split the story into sentences
def split_into_sentences(text):
    return sent_tokenize(text)

# Mapping the detected mood to actual file paths
background_music_files = {
    "Calm": "background_calm.mp3",
    "Happy": "background_happy.mp3",
    "Sad": "background_sad.mp3"
}

# Feedback list
feedback = []

# Custom CSS for visual enhancement
st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Arial', sans-serif;
        }
        .header, .footer {
            text-align: center;
            padding: 10px;
            background-color: #87CEFA;
            color: white;
        }
        .footer {
            margin-top: 20px;
        }
        .stButton button {
            background-color: #87CEFA;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
        .stTextInput > label {
            color: #333;
        }
        .stSlider > div {
            color: #87CEFA;
        }
        .stMarkdown > p {
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>AI-Powered Story Generator</h1></div>", unsafe_allow_html=True)

# Sidebar for window selection
window_option = st.sidebar.selectbox("Choose a window", ("Complete Story Generator", "Advanced Story Generator", "Custom Prompt Story Generator", "Generate Story Images"))

if window_option == "Complete Story Generator":
    st.header("Generate Stories from Images")
    st.write("Upload images to detect objects and analyze emotions, then generate a story.")

    model_choice = st.selectbox("Choose the model for object detection", ["Faster R-CNN", "Detectron2"])

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        all_keywords = []
        detected_emotions = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Processing...")

            if model_choice == "Faster R-CNN":
                detections = predict_faster_rcnn(image)
            else:
                detections = predict_detectron2(image)

            keywords = [detection['class'] for detection in detections if detection['score'] > 0.5]
            all_keywords.extend(keywords)

            img_array = np.array(image)
            emotion = detect_emotion(img_array)
            detected_emotions.append(emotion.capitalize() if emotion else "Neutral")

        keywords_str = ", ".join(all_keywords)
        st.write("Keywords for Story Generation:", keywords_str)

        selected_keywords = st.multiselect("Select 3 keywords for story generation", all_keywords, default=all_keywords[:3])

        if len(selected_keywords) != 3:
            st.error("Please select exactly 3 keywords.")
        else:
            if len(set(detected_emotions)) > 1:
                emotion = st.selectbox("Select an emotion for the story", list(set(detected_emotions)))
            else:
                emotion = detected_emotions[0] if detected_emotions else "Neutral"

            st.header("Story Generator")
            userpref = st.text_input("Enter a theme or genre (e.g., history, fantasy)", "history")

            if st.button("Generate Story"):
                with st.spinner("Generating your story..."):
                    payload = {
                        "keywords": selected_keywords,
                        "emotion": emotion,
                        "userpref": userpref
                    }

                    try:
                        API_URL_BASE = "http://127.0.0.1:8000"
                        response = requests.post(f"{API_URL_BASE}/generate_story", json=payload)

                        if response.status_code == 200:
                            story = response.json().get("story", "No story received.")
                            st.text_area("Generated Story", story, height=300)
                            st.success("Story generated successfully!")

                            st.header("Text to Audio Converter with Background Music")
                            language = st.selectbox("Select language", ("en", "es", "fr", "de", "it"))

                            sentences = split_into_sentences(story)
                            combined_audio = AudioSegment.silent(duration=0)

                            for i, sentence in enumerate(sentences):
                                if sentence.strip():
                                    sentence_filename = f"sentence_{i}.mp3"
                                    text_to_audio(sentence, language, sentence_filename)
                                    detected_mood = detect_mood(sentence)
                                    background_music_path = background_music_files.get(detected_mood, "background_calm.mp3")
                                    segment_output_path = f"output_segment_{i}.mp3"
                                    add_background_music_to_segment(sentence_filename, background_music_path, segment_output_path)

                                    segment_audio = AudioSegment.from_mp3(segment_output_path)
                                    combined_audio += segment_audio + AudioSegment.silent(duration=2000)

                                    os.remove(sentence_filename)
                                    os.remove(segment_output_path)

                            combined_output_path = "story_with_music.mp3"
                            with st.spinner("Generating audio..."):
                                combined_audio.export(combined_output_path, format="mp3")

                            audio_file = open(combined_output_path, "rb").read()
                            st.audio(audio_file, format="audio/mp3")
                            st.download_button(label="Download Audio", data=audio_file, file_name="story_with_music.mp3", mime="audio/mp3")

                            os.remove(combined_output_path)
                        else:
                            st.error(f"Error generating story: {response.status_code} - {response.text}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error generating story: {e}")

            # Feedback form
            st.header("Feedback")
            rating = st.slider("Rate the story (1-5)", 1, 5)
            comment = st.text_area("Leave a comment")

            if st.button("Submit Feedback"):
                feedback.append({"rating": rating, "comment": comment})
                st.success("Thank you for your feedback!")

elif window_option == "Advanced Story Generator":
    st.header("Advanced Story Generator")
    
    keywords = st.text_input("Enter keywords (comma separated)", "prince, frog, trees")
    emotion = st.text_input("Enter the desired emotion", "happy")
    userpref = st.text_input("Enter a theme or genre", "history")
    desc_caption = st.text_input("Enter a description caption", "A prince, a princess, and a historian venture into an enchanted forest filled with ancient trees to find a magical frog that holds a secret to happiness.")

    default_characters = [
        {"name": "Prince Alexander", "profession": "ruler and scholar"},
        {"name": "Princess Isabella", "profession": "adventurous princess"},
        {"name": "Prince Edward", "profession": "historian and childhood friend"}
    ]

    num_characters = st.number_input("Enter number of characters", min_value=1, max_value=10, value=3)

    characters = []
    for i in range(num_characters):
        default_name = default_characters[i]["name"] if i < len(default_characters) else f"Character {i+1} Name"
        default_profession = default_characters[i]["profession"] if i < len(default_characters) else f"Character {i+1} Profession"
        char_name = st.text_input(f"Enter name for character {i+1}", default_name)
        char_profession = st.text_input(f"Enter profession for character {i+1}", default_profession)
        characters.append({"name": char_name, "profession": char_profession})

    if st.button("Generate Story"):
        with st.spinner("Generating your story..."):
            progress_bar = st.progress(0)
            progress_bar.progress(10)

            keywords_list = [keyword.strip() for keyword in keywords.split(",")]
            payload = {
                "keywords": keywords_list,
                "emotion": emotion,
                "userpref": userpref,
                "num_characters": num_characters,
                "characters": characters,
                "desc_caption": desc_caption
            }
            progress_bar.progress(30)

            response = requests.post("http://127.0.0.1:8000/generate_enhanced_story", json=payload)
            progress_bar.progress(50)

            if response.status_code == 200:
                progress_bar.progress(70)
                story = response.json()["story"]
                progress_bar.progress(90)
                st.text_area("Generated Story", story, height=300)
                progress_bar.progress(100)
                st.success("Story generated successfully!")
            else:
                st.error("Error generating story: " + response.text)

            progress_bar.empty()

            # Feedback form
            st.header("Feedback")
            rating = st.slider("Rate the story (1-5)", 1, 5)
            comment = st.text_area("Leave a comment")

            if st.button("Submit Feedback"):
                feedback.append({"rating": rating, "comment": comment})
                st.success("Thank you for your feedback!")

elif window_option == "Custom Prompt Story Generator":
    st.header("Custom Prompt Story Generator")
    
    custom_prompt = st.text_area("Enter your custom prompt", "Generate a story that evokes a happy emotion. The story should feature a prince, a frog, and trees. Additionally, incorporate elements of history to enhance the narrative. Ensure the history aspects are seamlessly integrated and contribute to the overall happy tone of the story.")

    if st.button("Generate Story"):
        with st.spinner("Generating your story..."):
            progress_bar = st.progress(0)
            progress_bar.progress(10)

            payload = {"prompt": custom_prompt}
            progress_bar.progress(30)

            response = requests.post("http://127.0.0.1:8000/generate_custom_story", json=payload)
            progress_bar.progress(50)

            if response.status_code == 200:
                progress_bar.progress(70)
                story = response.json()["story"]
                progress_bar.progress(90)
                st.text_area("Generated Story", story, height=300)
                progress_bar.progress(100)
                st.success("Story generated successfully!")
            else:
                st.error("Error generating story: " + response.text)

            progress_bar.empty()

            # Feedback form
            st.header("Feedback")
            rating = st.slider("Rate the story (1-5)", 1, 5)
            comment = st.text_area("Leave a comment")

            if st.button("Submit Feedback"):
                feedback.append({"rating": rating, "comment": comment})
                st.success("Thank you for your feedback!")

elif window_option == "Generate Story Images":
    st.header("Generate Story Images")
    story = st.text_area("Enter your story here:", height=200)

    if st.button("Generate"):
        if story:
            with st.spinner("Sending request to server..."):
                response = requests.post("https://7b20-35-221-223-46.ngrok-free.app/generate", data={'story': story})
                if response.status_code == 200:
                    with st.spinner("Generating images and PDF..."):
                        response_data = response.json()
                        pdf_content = base64.b64decode(response_data['pdf'])

                        pdf_filename = "generated_story.pdf"
                        pdf_download_link = generate_download_link(pdf_content, pdf_filename, "application/pdf")
                        st.markdown(pdf_download_link, unsafe_allow_html=True)
                else:
                    st.error(f"Error generating images: {response.status_code} - {response.text}")
        else:
            st.error("Please enter a story.")

        # Feedback form
        st.header("Feedback")
        rating = st.slider("Rate the story (1-5)", 1, 5)
        comment = st.text_area("Leave a comment")

        if st.button("Submit Feedback"):
            feedback.append({"rating": rating, "comment": comment})
            st.success("Thank you for your feedback!")

st.markdown("<div class='footer'><p>Created by Mohamed Aziz Lahiani, Sivaprasad Puthumadthil Rameshan Nair, Rohithvishwa Vimalraj Sangeethapriya, Yazid Ben Madani</p></div>", unsafe_allow_html=True)
