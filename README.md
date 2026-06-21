# Speech Sentiment & Emotion Analyzer

An AI-powered web application that converts speech into text and analyzes the speaker's sentiment and emotions using state-of-the-art NLP models.

## Features

* Microphone Recording Support
* Audio File Upload (WAV, MP3, M4A)
* Speech-to-Text using OpenAI Whisper
* Sentiment Analysis (Positive / Negative)
* Emotion Detection (Joy, Sadness, Anger, Fear, Love, Surprise, Neutral)
* Keyword Extraction
* CSV History Storage
* Interactive Dashboard Charts
* PDF Report Generation

## Technologies Used

* Python
* Streamlit
* OpenAI Whisper
* Hugging Face Transformers
* YAKE Keyword Extraction
* Plotly
* Pandas
* ReportLab

## Workflow

Audio Input → Whisper Transcription → Keyword Extraction → Sentiment Analysis → Emotion Detection → CSV Storage → Dashboard Visualization → PDF Report

## Project Structure

speech-sentiment-analyzer/

├── app.py

├── history.csv

├── requirements.txt

├── README.md

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run application:

streamlit run app.py

## Future Improvements

* AI-generated summaries
* Multi-language support
* Cloud deployment
* Advanced analytics dashboard

## Live Demo

https://speech-sentiment-analyzer-saachi.streamlit.app/


## Author

Saachi Vij

B.Tech CCE 

Manipal University Jaipur
