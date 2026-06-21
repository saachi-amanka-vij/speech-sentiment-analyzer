from transformers import pipeline

print("Loading sentiment model...")

classifier = pipeline("sentiment-analysis")

result = classifier("I love this project")

print(result)