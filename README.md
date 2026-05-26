<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignLanguageTrack</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="./style.css">

    <!-- MediaPipe -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>

    <!-- ONNX Runtime Web -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

    <style>
        /* --- State Indicator --- */
        .state-indicator {
            position: absolute;
            top: 16px;
            left: 16px;
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: 20px;
            backdrop-filter: blur(4px);
            transition: background 0.2s, color 0.2s;
            z-index: 10;
        }
        .state-ready      { background: rgba(34, 197, 94, 0.18);  color: #22c55e; }
        .state-signing    { background: rgba(249, 115, 22, 0.18); color: #f97316; }
        .state-predicting { background: rgba(139, 92, 246, 0.18); color: #8b5cf6; }

        /* video + canvas overlay */
        .video-wrapper video,
        .video-wrapper canvas {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            object-fit: cover;
        }
        .video-wrapper video  { z-index: 1; }
        .video-wrapper canvas { z-index: 2; }

        /* loading overlay */
        .loading-msg {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Mono', monospace;
            font-size: 0.8rem;
            color: #888;
            z-index: 3;
            background: var(--surface);
        }

        /* --- How to Use button in header --- */
        #how-to-btn {
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 500;
            padding: 9px 18px;
            border-radius: 8px;
            border: 1.5px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s;
        }
        #how-to-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* --- How to Use modal overlay --- */
        .overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.6);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .overlay.active { display: flex; }

        .how-to-box {
            background: #fff;
            border-radius: 20px;
            padding: 40px 44px;
            max-width: 460px;
            width: 100%;
            box-shadow: 0 8px 40px rgba(0,0,0,0.15);
        }

        .how-to-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 22px;
            font-weight: 600;
            color: #111;
            margin-bottom: 6px;
        }

        .how-to-subtitle {
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #bbb;
            letter-spacing: 0.06em;
            margin-bottom: 28px;
        }

        .how-to-steps {
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-bottom: 32px;
        }

        .how-to-step {
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }

        .step-num {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--accent);
            background: var(--accent-light);
            border-radius: 99px;
            padding: 3px 10px;
            flex-shrink: 0;
            margin-top: 1px;
        }

        .step-text {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.9rem;
            color: #444;
            line-height: 1.5;
        }

        .step-text strong {
            color: #111;
            font-weight: 600;
        }

        #got-it-btn {
            width: 100%;
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            font-weight: 600;
            padding: 14px 20px;
            border-radius: 10px;
            border: none;
            background: var(--accent);
            color: #fff;
            cursor: pointer;
            transition: background 0.15s;
        }
        #got-it-btn:hover { background: #d06a1a; }
    </style>
</head>
<body>
    <header>
        <div class="logo">SignLanguage<span>Track</span></div>
        <div style="display:flex;align-items:center;gap:16px;">
            <button id="how-to-btn">How to Use</button>
            <div class="status">
                <span class="status-dot"></span>
                <span id="status-label">Loading…</span>
            </div>
        </div>
    </header>

    <main>
        <div class="video-panel">
            <div class="video-wrapper">
                <video id="webcam" autoplay playsinline muted style="display:none;"></video>
                <canvas id="overlay"></canvas>
                <div class="loading-msg" id="loading-msg">Starting camera…</div>
                <div class="state-indicator state-ready" id="state-indicator">Ready</div>
            </div>
        </div>

        <div class="info-panel">
            <div class="card sign-card">
                <div class="card-label">Detected Sign</div>
                <div class="sign-display" id="sign">—</div>
                <div class="confidence-row">
                    <span class="confidence-label">Confidence</span>
                    <span class="confidence-value" id="confidence-text"></span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar" id="confidence-bar"></div>
                </div>
            </div>

            <div class="card history-card">
                <div class="card-label">Recent Signs</div>
                <div class="history-list" id="history"></div>
            </div>

            <div class="card supported-card">
                <div class="card-label">Supported Signs</div>
                <div class="supported-list" id="supported-list"></div>
            </div>
        </div>
    </main>

    <!-- How to Use Modal -->
    <div class="overlay active" id="how-to-overlay">
        <div class="how-to-box">
            <div class="how-to-title">How to Use</div>
            <div class="how-to-subtitle">Real-time ASL sign recognition · 25 signs supported</div>
            <div class="how-to-steps">
                <div class="how-to-step">
                    <span class="step-num">1</span>
                    <span class="step-text"><strong>Allow camera access</strong> when prompted by your browser</span>
                </div>
                <div class="how-to-step">
                    <span class="step-num">2</span>
                    <span class="step-text"><strong>Perform an ASL sign</strong> clearly in front of your camera</span>
                </div>
                <div class="how-to-step">
                    <span class="step-num">3</span>
                    <span class="step-text"><strong>Lower your hand</strong> — the sign will be recognized instantly</span>
                </div>
                <div class="how-to-step">
                    <span class="step-num">4</span>
                    <span class="step-text"><strong>Repeat for the next sign</strong> — recent signs are tracked on the right</span>
                </div>
                <div class="how-to-step">
                    <span class="step-num">5</span>
                    <span class="step-text"><strong>Tap any sign</strong> in the supported list to see how it's performed on SignASL</span>
                </div>
            </div>
            <button id="got-it-btn">Start Signing</button>
        </div>
    </div>

    <script>
        const SIGN_URLS = {
            "hello":     "https://www.signasl.org/sign/hello",
            "yes":       "https://www.signasl.org/sign/yes",
            "no":        "https://www.signasl.org/sign/no",
            "nothing":   "https://www.signasl.org/sign/nothing",
            "thank you": "https://www.signasl.org/sign/thank-you",
            "please":    "https://www.signasl.org/sign/please",
            "eat":       "https://www.signasl.org/sign/eat",
            "drink":     "https://www.signasl.org/sign/drink",
            "water":     "https://www.signasl.org/sign/water",
            "more":      "https://www.signasl.org/sign/more",
            "apple":     "https://www.signasl.org/sign/apple",
            "mother":    "https://www.signasl.org/sign/mother",
            "father":    "https://www.signasl.org/sign/father",
            "book":      "https://www.signasl.org/sign/book",
            "walk":      "https://www.signasl.org/sign/walk",
            "cold":      "https://www.signasl.org/sign/cold",
            "hot":       "https://www.signasl.org/sign/hot",
            "black":     "https://www.signasl.org/sign/black",
            "carrot":    "https://www.signasl.org/sign/carrot",
            "go":        "https://www.signasl.org/sign/go",
            "day":       "https://www.signasl.org/sign/day",
            "break":     "https://www.signasl.org/sign/break",
            "cow":       "https://www.signasl.org/sign/cow",
            "monkey":    "https://www.signasl.org/sign/monkey",
            "draw":      "https://www.signasl.org/sign/draw",
            "type":      "https://www.signasl.org/sign/type",
        };
        const grid = document.getElementById('supported-list');
        Object.entries(SIGN_URLS).forEach(([sign, url]) => {
            const a = document.createElement('a');
            a.href = url;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.className = 'tag';
            a.textContent = sign.charAt(0).toUpperCase() + sign.slice(1);
            grid.appendChild(a);
        });
    </script>
    <script type="module" src="./main.js"></script>
</body>
</html>