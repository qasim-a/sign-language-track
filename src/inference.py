import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

SIGNS = ["hello", "yes", "no", "nothing", "thank you", "please", "eat", "drink", "water", "more", "apple", "mother"]
SEQUENCE_LENGTH = 30

class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.lstm = nn.LSTM(input_size=225, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, len(SIGNS))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = ASLModel()
model.load_state_dict(torch.load("models/asl_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

sequence = deque(maxlen=SEQUENCE_LENGTH)
# buffer smooths predictions by taking the most common result over last 5 frames
prediction_buffer = deque(maxlen=5)
current_sign = ""
confidence = 0.0

cap = cv2.VideoCapture(1)

no_hand_frames = 0
NO_HAND_RESET_THRESHOLD = 8

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_pose.Pose(min_detection_confidence=0.7) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)
        pose_result = pose.process(rgb)

        left_hand = np.zeros(63)
        right_hand = np.zeros(63)
        hand_detected = False

        if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
            hand_detected = True
            for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                label = handedness.classification[0].label
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                if label == "Left":
                    left_hand = landmarks
                else:
                    right_hand = landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if pose_result.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark]).flatten()
            mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            pose_landmarks = np.zeros(99)

        # reset sequence buffer when hand disappears to prevent blended sign sequences
        if not hand_detected:
            no_hand_frames += 1
            if no_hand_frames >= NO_HAND_RESET_THRESHOLD:
                sequence.clear()
                prediction_buffer.clear()
                current_sign = ""
                confidence = 0.0
        else:
            no_hand_frames = 0

        frame_landmarks = np.concatenate([left_hand, right_hand, pose_landmarks])
        sequence.append(frame_landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
                confidence = confidence.item()
                predicted = predicted.item()

            prediction_buffer.append(predicted)
            most_common = max(set(prediction_buffer), key=prediction_buffer.count)

            if confidence > 0.9 and SIGNS[most_common] != "nothing":
                current_sign = SIGNS[most_common]
            else:
                current_sign = ""

        display_text = f"Sign: {current_sign} ({confidence:.0%})" if current_sign else "Waiting..."
        color = (0, 255, 0) if current_sign else (0, 165, 255)
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imshow("ASL Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()