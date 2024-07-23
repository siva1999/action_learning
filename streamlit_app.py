import streamlit as st
import requests
import json
from datetime import datetime

# Assume user "Yazid" is logged in
if 'user_id' not in st.session_state:
    st.session_state.user_id = "yazid"  # Unique user identifier
if 'user_name' not in st.session_state:
    st.session_state.user_name = "epita"  # User's name

st.title('Text Generator')

if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'display_feedback' not in st.session_state:
    st.session_state.display_feedback = False

number_of_keywords = st.slider('How many keywords do you want to enter?', 3, 5)
keywords = []


sentiment = st.text_input("Enter sentiment", key='sentiment')
for i in range(number_of_keywords):
    keyword = st.text_input(f'Enter keyword {i+1}', key=f'keyword_{i}')
    if keyword:
        keywords.append(keyword)
if sentiment:
    keywords.append(sentiment)

if st.button('Generate Sentence'):
    st.session_state.keywords = keywords
    if keywords:
        prompt_text = ', '.join(keywords)
        response = requests.post('http://localhost:8000/generate-text/', json={"prompt": prompt_text})
        if response.status_code == 200:
            st.session_state.generated_text = response.json()['generated_text']
            st.session_state.display_feedback = True
        else:
            st.error("Failed to generate text. Status code: " + str(response.status_code))
    else:
        st.error("Please enter some keywords and sentiment.")

# Always display the generated text if available
if st.session_state.generated_text:
    st.write("Generated Sentence:", st.session_state.generated_text)

# Display the feedback section if enabled
if st.session_state.display_feedback:
    st.subheader("Feedback on Generated Text")
    rating = st.slider("Rate the output (1 is Poor, 5 is Excellent)", 1, 5, 3, key='rating')
    comment = st.text_area("Comment", key='comment')

    if st.button('Submit Feedback', key='submit_feedback'):
        feedback_data = {
            "user_id": st.session_state.user_id,
            "user_name": st.session_state.user_name,
            "timestamp": datetime.now().isoformat(),
            "generated_text": st.session_state.generated_text,
            "rating": rating,
            "comment": comment
        }

        # Write feedback to a JSON file
        with open('feedback.json', 'a') as f:
            json.dump(feedback_data, f)
            f.write('\n')  # Add newline to separate entries

        st.success("Thank you for your feedback!")
        st.session_state.display_feedback = False  # Reset feedback display state
