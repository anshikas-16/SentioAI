import torchaudio
import torch
from transformers import pipeline
import pyaudio
import wave

# Load model
print("Loading speech emotion model... please wait")
classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

def record_audio(filename="temp_audio.wav", duration=5):
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    print("✅ Recording done!")
    return filename

def analyze_speech_emotion(audio_file):
    result = classifier(audio_file)
    emotion = result[0]['label']
    score = round(result[0]['score'] * 100, 1)
    print(f"🎭 Detected Speech Emotion: {emotion} ({score}%)")
    return emotion, score

if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    emotion, score = analyze_speech_emotion(audio_file)