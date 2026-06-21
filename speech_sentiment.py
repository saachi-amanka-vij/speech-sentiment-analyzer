import whisper
from transformers import pipeline

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Loading Sentiment Model...")
classifier = pipeline("sentiment-analysis")

print("Transcribing audio...")

result = whisper_model.transcribe("samples/test.m4a")
text = result["text"]

print("\nTranscript:")
print(text)

sentiment = classifier(text)

print("\nSentiment:")
print(sentiment)