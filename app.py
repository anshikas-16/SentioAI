import gradio as gr
import whisper
import numpy as np
from deepface import DeepFace
from transformers import pipeline
from PIL import Image
import cv2

# ── Load models ────────────────────────────────────────────────
print("🔄 Loading SentioAI models... please wait")

whisper_model = whisper.load_model("tiny")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

speech_emotion_classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

print("✅ All models loaded!")

# ── Face emotion from uploaded/captured image ──────────────────
def analyze_face(image):
    if image is None:
        return "No face detected", None
    try:
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        result = DeepFace.analyze(img_bgr, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = round(result[0]['emotion'][emotion], 1)

        # Draw label on image
        cv2.putText(img_bgr, f"{emotion.upper()} ({confidence}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return emotion, Image.fromarray(img_rgb)
    except Exception as e:
        return "neutral", image

# ── Speech + text from recorded audio ─────────────────────────
def analyze_audio(audio):
    if audio is None:
        return "neutral", 0.0, "NEUTRAL", 0.0, "(No audio provided)"

    sample_rate, audio_data = audio

    # Save to wav
    import wave, struct
    filename = "temp_gradio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.astype(np.int16).tobytes())

    # Speech emotion
    speech_result = speech_emotion_classifier(filename)
    speech_emotion = speech_result[0]['label']
    speech_score = round(speech_result[0]['score'] * 100, 1)

    # Transcription + sentiment
    transcription = whisper_model.transcribe(filename)["text"].strip()
    if transcription:
        sent_result = sentiment_analyzer(transcription)
        sentiment = sent_result[0]['label']
        sent_score = round(sent_result[0]['score'] * 100, 1)
    else:
        sentiment = "NEUTRAL"
        sent_score = 0.0
        transcription = "(No speech detected)"

    return speech_emotion, speech_score, sentiment, sent_score, transcription

# ── Coaching feedback ──────────────────────────────────────────
def get_coaching_feedback(face_emotion, speech_emotion, sentiment):
    feedback = []
    if face_emotion in ["angry", "disgust", "fear"]:
        feedback.append("😬 Your face shows tension. Try relaxing your expressions.")
    elif face_emotion in ["happy", "surprise"]:
        feedback.append("😊 Great! Your face looks positive and engaging.")
    else:
        feedback.append("😐 Your face appears neutral. Try to smile more.")

    if speech_emotion in ["angry", "fearful"]:
        feedback.append("🎤 Your voice sounds stressed. Try speaking calmly and slowly.")
    elif speech_emotion in ["happy", "calm"]:
        feedback.append("🎤 Your voice tone is great — calm and confident!")
    else:
        feedback.append("🎤 Your voice is neutral. Add energy to engage better.")

    if sentiment == "POSITIVE":
        feedback.append("💬 Your words carry a positive message. Keep it up!")
    else:
        feedback.append("💬 Your words seem negative. Try reframing with positive language.")

    return "\n\n".join(feedback)

# ── Full analysis ──────────────────────────────────────────────
def full_analysis(image, audio, manual_text=""):
    # Face
    face_emotion, annotated_image = analyze_face(image)

    # Audio
    speech_emotion, speech_score, sentiment, sent_score, transcription = analyze_audio(audio)

    # If user typed manually, override transcription for sentiment
    if manual_text and manual_text.strip():
        transcription = manual_text.strip()
        sent_result = sentiment_analyzer(transcription)
        sentiment = sent_result[0]['label']
        sent_score = round(sent_result[0]['score'] * 100, 1)

    # Feedback
    feedback = get_coaching_feedback(face_emotion, speech_emotion, sentiment)

    report = f"""
## 🧠 SentioAI Analysis Report

| Channel | Result | Confidence |
|---|---|---|
| 😊 Face Emotion | {face_emotion.upper()} | — |
| 🎤 Speech Emotion | {speech_emotion.upper()} | {speech_score}% |
| 💬 Text Sentiment | {sentiment} | {sent_score}% |

---
**📝 You said:** {transcription}

---
## 🎯 Coaching Feedback
{feedback}
"""
    return annotated_image, report

# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(title="SentioAI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 SentioAI — Multimodal Emotional Intelligence Analyzer
    ### Analyze your face, voice & words in real-time to improve your communication
    ---
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Step 1 — Capture your face")
            image_input = gr.Image(
                sources=["webcam"],
                type="pil",
                label="Click to take a snapshot",
                height=300
            )

        with gr.Column():
            gr.Markdown("### 🎤 Step 2 — Record your voice")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Click mic to record, click stop when done",
            )
        
        with gr.Row():
            text_input = gr.Textbox(
                label="✍️ Step 3 — Type something (optional)",
                placeholder="e.g. I am feeling confident today and ready for this interview...",
                lines=3,
                info="If you type here, this text will be used for sentiment analysis instead of your speech transcription."
        )

    analyze_btn = gr.Button("🎬 Analyze Now!", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Face with Emotion Label")
            face_output = gr.Image(label="Detected Emotion", height=300)
        with gr.Column():
            report_output = gr.Markdown(
                value="*Your analysis report will appear here.*"
            )

    analyze_btn.click(
        fn=full_analysis,
        inputs=[image_input, audio_input, text_input],
        outputs=[face_output, report_output]
    )

    gr.Markdown("""
    ---
    *SentioAI — Built with DeepFace · Whisper · HuggingFace Transformers · Gradio*
    """)

if __name__ == "__main__":
    demo.launch()