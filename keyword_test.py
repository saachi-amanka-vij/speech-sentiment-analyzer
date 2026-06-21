import yake

text = """
Today I completed my speech sentiment analyzer project.
I learned Whisper, Hugging Face, Streamlit and Plotly.
"""

kw_extractor = yake.KeywordExtractor()

keywords = kw_extractor.extract_keywords(text)

for keyword, score in keywords[:5]:
    print(keyword)