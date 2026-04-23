# Sign Language Tracker

A real-time ASL recognition system that uses your webcam to identify hand signs and display the corresponding English word.

## What it does

Point your webcam at your hand, perform one of the supported ASL signs, and the app will show you what sign it thinks you're making along with a confidence score. Built mainly for beginners learning ASL who want quick feedback.

v1 supports around 10-20 signs. No sentence translation, just individual sign recognition.

## How it works

1. Collect webcam data for each sign
2. Extract hand landmarks using MediaPipe
3. Train a sequence classifier with PyTorch
4. Run live predictions with smoothing so output stays stable

## Stack

- MediaPipe, PyTorch, OpenCV, Python 3.10+

## Status

Work in progress

## Setup

```bash
git clone https://github.com/qasim-a/sign-language-track.git
cd sign-language-track
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```