import streamlit as st
import whisper
from transformers import pipeline
import tempfile

st.title("🎤 Speech Sentiment Analyzer")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file is not None:

    st.success("File uploaded successfully!")

    if st.button("Analyze"):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Load Whisper
        st.write("Loading Whisper model...")
        whisper_model = whisper.load_model("base")

        # Convert speech to text
        st.write("Transcribing audio...")
        result = whisper_model.transcribe(temp_path)

        transcript = result["text"]

        # Show transcript
        st.subheader("Transcript")
        st.write(transcript)

        # Load AI models
        st.write("Analyzing sentiment and emotion...")

        sentiment_classifier = pipeline("sentiment-analysis")

        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )

        # Run predictions
        sentiment = sentiment_classifier(transcript)
        emotion = emotion_classifier(transcript)

        # Sentiment Results
        st.subheader("Sentiment")
        st.write(sentiment[0]["label"])

        st.subheader("Sentiment Confidence")
        st.write(f"{sentiment[0]['score']:.2%}")

        # Emotion Results
        st.subheader("Emotion")
        st.write(emotion[0]["label"].upper())

        st.subheader("Emotion Confidence")
        st.write(f"{emotion[0]['score']:.2%}")