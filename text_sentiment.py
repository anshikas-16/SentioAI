import whisper
from transformers import pipeline

print("Loading models... please wait")

# Load Whisper for speech-to-text
whisper_model = whisper.load_model("tiny")

# Load sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def transcribe_audio(audio_file="temp_audio.wav"):
    print("📝 Transcribing your speech...")
    result = whisper_model.transcribe(audio_file)
    text = result["text"].strip()
    print(f"📄 You said: {text}")
    return text

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    score = round(result[0]['score'] * 100, 1)
    print(f"💬 Sentiment: {sentiment} ({score}%)")
    return sentiment, score

if __name__ == "__main__":
    # Uses the same temp_audio.wav recorded by speech_emotion.py
    text = transcribe_audio("temp_audio.wav")
    if text:
        sentiment, score = analyze_sentiment(text)
    else:
        print("No speech detected in audio.")