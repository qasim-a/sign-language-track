# SignLanguageTrack

A real-time ASL recognition system that uses a webcam to identify hand signs and display the corresponding English word through a web interface.

## What it does

Point your webcam at your hand, perform one of the supported ASL signs, and the app will display the recognized sign along with a confidence bar. Built for beginners learning ASL who want quick feedback on their signing.

v1 supports 3 signs. No sentence translation, just individual sign recognition.

## How it works

1. **Collect** — record hand gesture sequences via webcam, MediaPipe extracts 63 hand landmark coordinates per frame
2. **Preprocess** — sequences are labeled and saved into a dataset
3. **Train** — an LSTM model is trained on the sequences to classify each gesture
4. **Run** — Flask serves a live web interface that streams webcam feed and displays real-time predictions

## Stack

- MediaPipe — hand landmark extraction
- PyTorch — LSTM model training and inference
- OpenCV — webcam capture
- Flask — local web server
- Python 3.10+

## Supported Signs (v1)

Hello, Yes, No

## Setup

```bash
git clone https://github.com/qasim-a/sign-language-track.git
cd sign-language-track
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Collect data:**
```bash
python src/collect_data.py
```

**Preprocess:**
```bash
python src/preprocess.py
```

**Train:**
```bash
python src/train.py
```

**Run the app:**
```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Project Status

v1 in progress — core pipeline complete, expanding sign set next.