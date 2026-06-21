from transformers import pipeline

print("Loading summarizer...")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = """
Today I completed my speech sentiment analyzer project.
I learned Whisper, Hugging Face, Streamlit and Plotly.
The project can now transcribe speech, detect sentiment,
detect emotions, store history and show dashboards.
"""

summary = summarizer(
    text,
    max_length=50,
    min_length=10,
    do_sample=False
)

print("\nSummary:")
print(summary[0]["summary_text"])