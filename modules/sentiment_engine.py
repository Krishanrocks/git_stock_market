from textblob import TextBlob
import re

def compute_sentiment(text):
    if not text or not isinstance(text, str):
        return 0.0
    try:
        blob = TextBlob(re.sub(r"\s+", " ", text))
        return float(blob.sentiment.polarity)
    except:
        return 0.0
