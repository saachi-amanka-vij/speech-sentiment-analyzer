import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Transcribing audio...")

result = model.transcribe("samples/test.m4a")

print("\nTranscript:")
print(result["text"])
