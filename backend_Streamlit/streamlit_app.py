# import streamlit as st
# import requests
# import json
#
# st.title("Loan Docs Assistant")
#
# # Input from user
# input_text = st.text_area("Enter loan document text here:")
#
# # API base URL
# API_URL = "http://127.0.0.1:8501"
#
# # Buttons to call FastAPI endpoints
# if st.button("Translate"):
#     response = requests.post(f"{API_URL}/translate", json={"text": input_text})
#     st.write(response.json()["result"])
#
# if st.button("Summarize"):
#     response = requests.post(f"{API_URL}/summarize", json={"text": input_text})
#     st.write(response.json()["result"])
#
# if st.button("TTS"):
#     response = requests.post(f"{API_URL}/tts", json={"text": input_text})
#     st.write(response.json()["result"])
#
# if st.button("Math Explain"):
#     response = requests.post(f"{API_URL}/math_explain", json={"text": input_text})
#     st.write(response.json()["result"])
import streamlit as st
import requests

st.title("Loan Docs Assistant - Streamlit UI")

# Input from user
input_text = st.text_area("Enter loan document text here:")

# API base URL
API_URL = "http://127.0.0.1:8501"

# Translate button
if st.button("Translate"):
    if input_text.strip() != "":
        response = requests.post(f"{API_URL}/translate", json={"text": input_text})
        st.write(response.json()["result"])
    else:
        st.warning("Please enter some text.")

# Summarize button
if st.button("Summarize"):
    if input_text.strip() != "":
        response = requests.post(f"{API_URL}/summarize", json={"text": input_text})
        st.write(response.json()["result"])
    else:
        st.warning("Please enter some text.")

# TTS button
if st.button("Voice-Assistance"):
    if input_text.strip() != "":
        response = requests.post(f"{API_URL}/tts", json={"text": input_text})
        st.write(response.json()["result"])
    else:
        st.warning("Please enter some text.")

# Math Explain button
if st.button("Math Explain"):
    if input_text.strip() != "":
        response = requests.post(f"{API_URL}/math_explain", json={"text": input_text})
        st.write(response.json()["result"])
    else:
        st.warning("Please enter some text.")
