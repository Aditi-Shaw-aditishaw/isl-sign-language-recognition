import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time

# Load model and tools
model = joblib.load("model/gesture_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Setup voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 140)
engine.setProperty('volume', 1.0)

# 🔊 Try switching to a clearer/louder voice
voices = engine.getProperty('voices')
for voice in voices:
    if "Zira" in voice.name:  # Optional: use a clearer voice on Windows
        engine.setProperty('voice', voice.id)

def speak(text):
    engine.stop()             # Stop any ongoing speech
    engine.say(text)
    engine.runAndWait()
    time.sleep(0.3)           # Delay to prevent clipping

# MediaPipe setup
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
print("🟢 ISL Sentence Mode (SPACE=add word, ENTER=speak sentence, C=clear word, R=reset sentence)")

# Buffers
letter_buffer = ""
sentence_buffer = []
last_prediction = ""
last_letter_added = ""
repeat_count = 0
buffer_count = 0
BUFFER_LIMIT = 3

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) in [1, 2]:
        hand_landmarks = sorted(result.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        row = []

        for hand in hand_landmarks[:2]:
            wrist = hand.landmark[0]
            for lm in hand.landmark:
                row.append(round(lm.x - wrist.x, 5))
                row.append(round(lm.y - wrist.y, 5))

        while len(row) < 84:
            row.append(0.0)

        X_input = scaler.transform([row])
        pred = model.predict(X_input)[0]
        gesture = label_encoder.inverse_transform([pred])[0]

        if gesture == last_prediction:
            buffer_count += 1
            if buffer_count >= BUFFER_LIMIT:
                if gesture == last_letter_added:
                    repeat_count += 1
                else:
                    repeat_count = 1

                if repeat_count <= 2:
                    letter_buffer += gesture
                    last_letter_added = gesture
                    speak(gesture)

                buffer_count = 0
        else:
            buffer_count = 0
            last_prediction = gesture

        cv2.putText(frame, f"Detected: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    # Show word + sentence
    cv2.putText(frame, f"Word: {letter_buffer}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    full_sentence = " ".join(sentence_buffer)
    cv2.putText(frame, f"Sentence: {full_sentence}", (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 200), 2)

    cv2.imshow("ISL Sentence Builder (Voice Fixed)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE = add word
        if letter_buffer:
            sentence_buffer.append(letter_buffer)
            print(f"➕ Added word: {letter_buffer}")
            speak(letter_buffer)
            letter_buffer = ""
            last_letter_added = ""
            repeat_count = 0
            buffer_count = 0
    elif key == 13:  # ENTER = speak sentence
        if sentence_buffer:
            print(f"🗣️ Full Sentence: {full_sentence}")
            speak(full_sentence)
    elif key == ord('c'):  # Clear current word
        letter_buffer = ""
        repeat_count = 0
        buffer_count = 0
        print("🧹 Cleared current word buffer")
    elif key == ord('r'):  # Reset full sentence
        sentence_buffer = []
        print("🔁 Reset full sentence buffer")

cap.release()
cv2.destroyAllWindows()
