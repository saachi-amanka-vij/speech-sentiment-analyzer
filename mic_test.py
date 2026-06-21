import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.title("🎤 Microphone Test")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True
)

if audio:
    st.success("Recording captured successfully!")
    st.write(audio.keys())