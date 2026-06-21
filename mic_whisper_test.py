import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile

st.title("🎤 Speech To Text Test")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True
)

if audio:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio["bytes"])
        temp_path = temp_audio.name

    st.write("Loading Whisper...")

    model = whisper.load_model("base")

    st.write("Transcribing...")

    result = model.transcribe(temp_path)

    st.subheader("Transcript")
    st.write(result["text"])