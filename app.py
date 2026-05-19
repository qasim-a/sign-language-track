import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from flask import Flask, render_template, Response, jsonify
import threading
import time

app = Flask(__name__)

SIGNS = ["hello", "yes", "no", "nothing", "thank you", "please", "eat", "drink", "water", "more", "apple", "mother", "father", "book", "walk", "cold", "hot", "me", "you", "black", "carrot", "go", "night", "day", "break", "cow", "monkey"]
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

# initialize mediapipe outside generate_frames so they persist across frames
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# rolling frame buffer — fills during SIGNING state, consumed once on transition to PREDICTING
sequence = deque(maxlen=SEQUENCE_LENGTH)

current_prediction = {
    "sign": "",
    "confidence": 0.0,
    "history": [],
    "state": "ready"    # "ready" | "signing" | "predicting"
}

# lock prevents race conditions between the video stream and prediction threads
lock = threading.Lock()

cap = cv2.VideoCapture(0)
print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

def run_inference(frames):
    """Run the LSTM once on a completed sequence. Always returns a result — no confidence threshold.
    Only suppresses the 'nothing' class since that's not a real sign."""
    if len(frames) < SEQUENCE_LENGTH:
        # pad with zeros at the front if we didn't get a full 30 frames
        pad = [np.zeros(225)] * (SEQUENCE_LENGTH - len(frames))
        frames = pad + list(frames)

    input_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        confidence = confidence.item()
        predicted = predicted.item()

    sign = SIGNS[predicted]

    # suppress "nothing" — that class absorbs non-sign frames, not a user-facing result
    if sign == "nothing":
        return None, round(confidence * 100)

    return sign, round(confidence * 100)

def generate_frames():
    # --- state machine ---
    # READY      : waiting for hand to appear
    # SIGNING    : hand detected, accumulating frames
    # PREDICTING : hand gone, inference fired, result on screen
    #              auto-resets to READY after 1.5s or immediately on new hand
    state = "ready"
    no_hand_frames = 0
    NO_HAND_TRIGGER = 8     # frames of absent hand before sign is considered complete
    predict_time = None     # set on first frame of predicting state

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)
        pose_result = pose.process(rgb)

        # --- extract landmarks ---
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

        frame_landmarks = np.concatenate([left_hand, right_hand, pose_landmarks])

        # --- state transitions ---
        if state == "ready":
            if hand_detected:
                state = "signing"
                no_hand_frames = 0
                sequence.clear()
                sequence.append(frame_landmarks)
                with lock:
                    current_prediction["state"] = "signing"
                    current_prediction["sign"] = ""
                    current_prediction["confidence"] = 0.0

        elif state == "signing":
            if hand_detected:
                no_hand_frames = 0
                sequence.append(frame_landmarks)
            else:
                no_hand_frames += 1
                # keep appending during brief occlusions so motion isn't cut short
                sequence.append(frame_landmarks)

                if no_hand_frames >= NO_HAND_TRIGGER:
                    # sign is complete — run inference once
                    state = "predicting"
                    predict_time = None
                    sign, confidence = run_inference(list(sequence))

                    with lock:
                        current_prediction["state"] = "predicting"
                        if sign:
                            current_prediction["sign"] = sign
                            current_prediction["confidence"] = confidence
                            history = current_prediction["history"]
                            if not history or history[-1] != sign:
                                history.append(sign)
                                if len(history) > 5:
                                    history.pop(0)
                        else:
                            # "nothing" class predicted — show blank result
                            current_prediction["sign"] = ""
                            current_prediction["confidence"] = 0.0

                    sequence.clear()

        elif state == "predicting":
            if predict_time is None:
                predict_time = time.time()

            # auto-reset to ready after 1.5s, or immediately if a new hand appears
            if time.time() - predict_time > 1.5 or hand_detected:
                state = "ready"
                predict_time = None
                with lock:
                    current_prediction["state"] = "ready"
                    current_prediction["sign"] = ""
                    current_prediction["confidence"] = 0.0

        # --- video overlay ---
        overlay_color = {
            "ready":      (100, 200, 100),
            "signing":    (50,  180, 255),
            "predicting": (80,  80,  220),
        }[state]

        overlay_text = {
            "ready":      "READY",
            "signing":    f"SIGNING  {len(sequence)}/{SEQUENCE_LENGTH}",
            "predicting": "DONE",
        }[state]

        

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    with lock:
        return jsonify(current_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)