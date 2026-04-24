import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

SIGNS = ["hello", "yes", "no", "nothing"]
SEQUENCE_LENGTH = 30

class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = ASLModel()
model.load_state_dict(torch.load("models/asl_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=5)
current_sign = ""
confidence = 0.0

cap = cv2.VideoCapture(1)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark]).flatten()
        else:
            landmarks = np.zeros(63)

        sequence.append(landmarks)

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

            if confidence > 0.9:
                current_sign = SIGNS[most_common]
                if current_sign == "nothing":
                    current_sign = ""

        display_text = f"Sign: {current_sign} ({confidence:.0%})" if current_sign else "Waiting..."
        color = (0, 255, 0) if current_sign else (0, 165, 255)
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imshow("ASL Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()