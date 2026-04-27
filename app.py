import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from flask import Flask, render_template, Response, jsonify
import threading

app = Flask(__name__)

SIGNS = ["hello", "yes", "no", "nothing", "thank you", "please"]
SEQUENCE_LENGTH = 30

class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, len(SIGNS))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = ASLModel()
model.load_state_dict(torch.load("models/asl_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

sequence = deque(maxlen=SEQUENCE_LENGTH)
# buffer smooths predictions by taking the most common result over last 5 frames
prediction_buffer = deque(maxlen=5)

current_prediction = {"sign": "", "confidence": 0.0, "history": []}
# lock prevents race conditions between the video stream and prediction threads
lock = threading.Lock()

cap = cv2.VideoCapture(1)

def generate_frames():
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

            with lock:
                if confidence > 0.9 and SIGNS[most_common] != "nothing":
                    sign = SIGNS[most_common]
                    current_prediction["sign"] = sign
                    current_prediction["confidence"] = round(confidence * 100)
                    history = current_prediction["history"]
                    if not history or history[-1] != sign:
                        history.append(sign)
                        if len(history) > 5:
                            history.pop(0)
                else:
                    current_prediction["sign"] = ""
                    current_prediction["confidence"] = 0.0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# streams webcam frames as multipart JPEG to the browser
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    with lock:
        return jsonify(current_prediction)

if __name__ == '__main__':
    app.run(debug=False)