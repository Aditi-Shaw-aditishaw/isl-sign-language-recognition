import cv2
import mediapipe as mp
import csv
import os

# Setup
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Input from user
label = input("Enter the gesture label (e.g., A, B, HELLO): ").upper()
filename = f"{label}_gesture_data.csv"
output_dir = "gesture_dataset"
os.makedirs(output_dir, exist_ok=True)

# Settings
data = []
target_samples = 200
count = 0

print(f"📦 Collecting samples for label: {label}")

while count < target_samples:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    row = []
    if results.multi_hand_landmarks:
        hand_landmarks = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

        for hand in hand_landmarks[:2]:  # Up to 2 hands
            wrist = hand.landmark[0]
            for lm in hand.landmark:
                row.append(round(lm.x - wrist.x, 5))
                row.append(round(lm.y - wrist.y, 5))

        # If only 1 hand → pad to 84 features
        while len(row) < 84:
            row.append(0.0)

        row.append(label)
        data.append(row)
        count += 1

        # Draw hand landmarks
        for hand in hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

        print(f"✅ Captured: {count}/{target_samples}")

    # Display
    cv2.putText(frame, f"Label: {label} ({count}/{target_samples})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collection (1 or 2 Hands)", frame)

    # ESC to stop early
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save to CSV
header = [f'x{i}' if i % 2 == 0 else f'y{i//2}' for i in range(42)] * 2 + ['label']
file_path = os.path.join(output_dir, filename)

with open(file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f"📁 Saved: {file_path}")
cap.release()
cv2.destroyAllWindows()
