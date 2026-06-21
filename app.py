import streamlit as st
import whisper
from transformers import pipeline
import tempfile
from streamlit_mic_recorder import mic_recorder

import csv
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import yake

from reportlab.pdfgen import canvas

st.set_page_config(page_title="Speech Sentiment Analyzer")

st.title("🎤 Speech Sentiment & Emotion Analyzer")

# -----------------------------
# SESSION STATE FOR MICROPHONE
# -----------------------------

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# -----------------------------
# INPUT METHODS
# -----------------------------

st.header("Choose Input Method")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a"]
)

st.write("### OR")

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True
)

if audio:
    st.session_state.audio_bytes = audio["bytes"]
    st.success("✅ Recording captured successfully!")

# -----------------------------
# DETERMINE AUDIO SOURCE
# -----------------------------

temp_path = None

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

elif st.session_state.audio_bytes:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(st.session_state.audio_bytes)
        temp_path = temp_audio.name

# -----------------------------
# ANALYZE BUTTON
# -----------------------------

if temp_path:

    if st.button("🚀 Analyze"):

        # Load Whisper
        with st.spinner("Loading Whisper Model..."):
            whisper_model = whisper.load_model("base")

        # Speech to Text
        with st.spinner("Transcribing Audio..."):
            result = whisper_model.transcribe(temp_path)

        transcript = result["text"]

        st.subheader("📝 Transcript")
        st.write(transcript)

        # -----------------------------
        # KEYWORD EXTRACTION
        # -----------------------------

        keywords = []

        try:
            kw_extractor = yake.KeywordExtractor()

            keywords = kw_extractor.extract_keywords(transcript)

            st.subheader("🔑 Keywords")

            for keyword, score in keywords[:5]:
                st.write(f"• {keyword}")

        except Exception as e:
            st.warning(f"Keyword extraction error: {e}")

        # -----------------------------
        # SENTIMENT MODEL
        # -----------------------------

        with st.spinner("Loading Sentiment Model..."):
            sentiment_classifier = pipeline(
                "sentiment-analysis"
            )

        # -----------------------------
        # EMOTION MODEL
        # -----------------------------

        with st.spinner("Loading Emotion Model..."):
            emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )

        # Predictions

        sentiment = sentiment_classifier(transcript)
        emotion = emotion_classifier(transcript)

        sentiment_label = sentiment[0]["label"]
        sentiment_score = sentiment[0]["score"]

        emotion_label = emotion[0]["label"]
        emotion_score = emotion[0]["score"]

        # -----------------------------
        # SAVE TO CSV
        # -----------------------------

        file_exists = (
            os.path.isfile("history.csv")
            and os.path.getsize("history.csv") > 0
        )

        with open("history.csv", "a", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "Transcript",
                    "Sentiment",
                    "Emotion"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                transcript,
                sentiment_label,
                emotion_label
            ])

        st.success("✅ Analysis saved to history.csv")

        # -----------------------------
        # SENTIMENT RESULTS
        # -----------------------------

        st.subheader("😊 Sentiment")

        if sentiment_label == "POSITIVE":
            st.success(sentiment_label)
        else:
            st.error(sentiment_label)

        st.write(f"Confidence: {sentiment_score:.2%}")
        st.progress(float(sentiment_score))

        # -----------------------------
        # EMOTION RESULTS
        # -----------------------------

        st.subheader("🎭 Emotion")

        emotion_emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😲",
            "love": "❤️",
            "neutral": "😐"
        }

        emoji = emotion_emojis.get(emotion_label, "🤖")

        st.info(f"{emoji} {emotion_label.upper()}")

        st.write(f"Confidence: {emotion_score:.2%}")
        st.progress(float(emotion_score))

        # -----------------------------
        # PDF REPORT GENERATION
        # -----------------------------

        pdf_file = "speech_report.pdf"

        c = canvas.Canvas(pdf_file)

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 800, "Speech Analysis Report")

        c.setFont("Helvetica", 12)

        c.drawString(
            50,
            770,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        c.drawString(50, 730, "Transcript:")
        c.drawString(50, 710, transcript[:80])

        c.drawString(50, 670, f"Sentiment: {sentiment_label}")
        c.drawString(50, 650, f"Emotion: {emotion_label}")

        c.drawString(50, 610, "Keywords:")

        y = 590

        if keywords:
            for keyword, score in keywords[:5]:
                c.drawString(70, y, f"- {keyword}")
                y -= 20
        else:
            c.drawString(70, y, "No keywords available")

        c.save()

        with open(pdf_file, "rb") as pdf:

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf,
                file_name="speech_report.pdf",
                mime="application/pdf"
            )

# -----------------------------
# HISTORY TABLE
# -----------------------------

st.subheader("📊 Analysis History")

if os.path.exists("history.csv"):

    history = pd.read_csv("history.csv")

    st.dataframe(history)

    # -----------------------------
    # DASHBOARD CHARTS
    # -----------------------------

    st.subheader("📈 Sentiment Distribution")

    sentiment_counts = history["Sentiment"].value_counts()

    sentiment_fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution"
    )

    st.plotly_chart(sentiment_fig, use_container_width=True)

    st.subheader("🎭 Emotion Distribution")

    emotion_counts = history["Emotion"].value_counts()

    emotion_fig = px.bar(
        x=emotion_counts.index,
        y=emotion_counts.values,
        title="Emotion Distribution"
    )

    st.plotly_chart(emotion_fig, use_container_width=True)

else:

    st.info("No analysis history found.")