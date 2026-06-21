from transformers import pipeline

print("Loading Emotion Model...")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

text = "I am extremely sad and disappointed."

result = emotion_classifier(text)

print(result)