import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SIGNS = ["hello", "yes", "no"]
SEQUENCES = 30
FRAMES = 30

for sign in SIGNS:
    os.makedirs(os.path.join("data", sign), exist_ok=True)

cap = cv2.VideoCapture(1)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    for sign in SIGNS:

        # long countdown when switching to a new sign
        for countdown in range(5, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f"Next sign: {sign.upper()}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Get ready in {countdown}...", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Collecting", frame)
            cv2.waitKey(1000)

        for seq in range(SEQUENCES):
            frames = []

            # short countdown between sequences
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, f"Sign: {sign.upper()} | Seq {seq + 1}/{SEQUENCES}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Get ready: {countdown}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Collecting", frame)
                cv2.waitKey(1000)

            for frame_num in range(FRAMES):
                ret, frame = cap.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark]).flatten()
                else:
                    landmarks = np.zeros(63)

                frames.append(landmarks)

                cv2.putText(frame, f"Sign: {sign.upper()} | Seq {seq + 1}/{SEQUENCES}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Recording... Frame {frame_num + 1}/{FRAMES}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Collecting", frame)
                cv2.waitKey(1)

            existing = len(os.listdir(os.path.join("data", sign)))
            np.save(os.path.join("data", sign, f"seq_{existing + seq}.npy"), np.array(frames))
            print(f"Saved sequence {seq + 1}/{SEQUENCES} for '{sign}'")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")