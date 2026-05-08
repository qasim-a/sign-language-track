import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

SIGNS = ["hello", "yes", "no", "eat", "please"]
SEQUENCES = 20
FRAMES = 30

for sign in SIGNS:
    os.makedirs(os.path.join("data", sign), exist_ok=True)

cap = cv2.VideoCapture(1)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_pose.Pose(min_detection_confidence=0.7) as pose:

    for sign in SIGNS:

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

                hand_result = hands.process(rgb)
                pose_result = pose.process(rgb)

                # --- hand landmarks (left and right, 63 each, zeros if absent) ---
                left_hand = np.zeros(63)
                right_hand = np.zeros(63)

                if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                    for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                        label = handedness.classification[0].label  # "Left" or "Right"
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        if label == "Left":
                            left_hand = landmarks
                        else:
                            right_hand = landmarks
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- pose landmarks (33 points, zeros if absent) ---
                if pose_result.pose_landmarks:
                    pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark]).flatten()
                    mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    pose_landmarks = np.zeros(99)

                # combine: left hand (63) + right hand (63) + pose (99) = 225 per frame
                frame_landmarks = np.concatenate([left_hand, right_hand, pose_landmarks])
                frames.append(frame_landmarks)

                cv2.putText(frame, f"Sign: {sign.upper()} | Seq {seq + 1}/{SEQUENCES}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Recording... Frame {frame_num + 1}/{FRAMES}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Collecting", frame)
                cv2.waitKey(1)

            # count existing files to avoid overwriting previous recordings
            existing = len(os.listdir(os.path.join("data", sign)))
            np.save(os.path.join("data", sign, f"seq_{existing + seq}.npy"), np.array(frames))
            print(f"Saved sequence {seq + 1}/{SEQUENCES} for '{sign}'")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")