import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)  # opens your webcam

print("SentioAI - Face Emotion Module Running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = round(result[0]['emotion'][emotion], 1)

        # Draw label on frame
        cv2.putText(frame, f"{emotion} ({confidence}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, "Detecting...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("SentioAI - Face Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()