import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ── Configuration ─────────────────────────────────────────────────────────────

# Set to True when recording messy/entry sequences, False for clean sequences.
# Filenames will automatically reflect the mode:
#   clean: seq_clean_000.npy, seq_clean_001.npy ...
#   messy: seq_messy_000.npy, seq_messy_001.npy ...
MESSY = 

SIGNS = ["type"]
SEQUENCES = 50    # number of sequences to record per sign in this session
FRAMES = 30        # frames per sequence — must match model input (do not change)
CAMERA = 0          # camera index — change to 1 if using external webcam

# ─────────────────────────────────────────────────────────────────────────────

PREFIX = "seq_messy" if MESSY else "seq_clean"
MODE_LABEL = "MESSY" if MESSY else "CLEAN"

for sign in SIGNS:
    os.makedirs(os.path.join("data", sign), exist_ok=True)

cap = cv2.VideoCapture(CAMERA)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_pose.Pose(min_detection_confidence=0.7) as pose:

    for sign in SIGNS:

        for countdown in range(5, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f"[{MODE_LABEL}] Next: {sign.upper()}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Get ready in {countdown}...", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Collecting", frame)
            cv2.waitKey(1000)

        # count existing sequences of this type to find the next index
        folder = os.path.join("data", sign)
        existing = len([f for f in os.listdir(folder) if f.startswith(PREFIX)])

        for seq in range(SEQUENCES):
            frames = []
            seq_index = existing + seq

            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, f"[{MODE_LABEL}] {sign.upper()} | Seq {seq_index:03d}", (30, 50),
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
                        label = handedness.classification[0].label
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        if label == "Left":
                            left_hand = landmarks
                        else:
                            right_hand = landmarks
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- pose landmarks (33 points × 3 = 99, zeros if absent) ---
                if pose_result.pose_landmarks:
                    pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark]).flatten()
                    mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    pose_landmarks = np.zeros(99)

                # combine: left hand (63) + right hand (63) + pose (99) = 225 per frame
                frame_landmarks = np.concatenate([left_hand, right_hand, pose_landmarks])
                frames.append(frame_landmarks)

                cv2.putText(frame, f"[{MODE_LABEL}] {sign.upper()} | Seq {seq_index:03d}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Recording... Frame {frame_num + 1}/{FRAMES}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Collecting", frame)
                cv2.waitKey(1)

            filename = f"{PREFIX}_{seq_index:03d}.npy"
            np.save(os.path.join(folder, filename), np.array(frames))
            print(f"Saved {filename} for '{sign}' [{MODE_LABEL}]")

cap.release()
cv2.destroyAllWindows()
print(f"Data collection complete. Mode: {MODE_LABEL}")