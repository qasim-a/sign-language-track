/**
 * inference.js
 * Browser-side replacement for app.py.
 * Handles webcam, MediaPipe landmark extraction,
 * ONNX inference, and the READY→SIGNING→PREDICTING state machine.
 */

const SIGNS = [
    "hello", "yes", "no", "nothing", "thank you", "please", "eat", "drink",
    "water", "more", "apple", "mother", "father", "book", "walk", "cold",
    "hot", "black", "carrot", "go", "day", "break", "cow", "monkey", "draw", "type"
];

const SEQUENCE_LENGTH  = 30;
const NO_HAND_TRIGGER  = 5;
const PREDICT_RESET_MS = 1500;
const MODEL_PATH       = "./asl_model.onnx";

let predictionState = {
    sign: "", confidence: 0, state: "ready", history: []
};

export function getPrediction() {
    return { ...predictionState };
}

// internal state machine variables
let onnxSession  = null;
let smState      = "ready";
let sequence     = [];
let noHandFrames = 0;
let predictTimer = null;

/**
 * startInference — initialises webcam, MediaPipe, ONNX and starts the loop.
 * Resolves when everything is ready and the camera is running.
 */
export async function startInference(videoEl, canvasEl, onStateChange = () => {}) {
    // configure ONNX Runtime wasm paths
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    
    // load ONNX model
    onnxSession = await ort.InferenceSession.create(MODEL_PATH);
    console.log("[inference] ONNX model loaded");

    // 2. start webcam
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" }
    });
    videoEl.srcObject = stream;
    await new Promise(resolve => { videoEl.onloadedmetadata = resolve; });
    videoEl.play();
    console.log("[inference] Webcam started");

    // 3. MediaPipe Hands
    const hands = new Hands({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    hands.setOptions({
        maxNumHands:            2,
        modelComplexity:        1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence:  0.5
    });

    // 4. MediaPipe Pose
    const pose = new Pose({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`
    });
    pose.setOptions({
        modelComplexity:        1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence:  0.5
    });

    const ctx = canvasEl.getContext("2d");
    let latestHandResult = null;
    let latestPoseResult = null;

    hands.onResults(r => { latestHandResult = r; });
    pose.onResults(r  => { latestPoseResult = r; });

    // 5. Camera utility — drives the per-frame loop
    const camera = new Camera(videoEl, {
        onFrame: async () => {
            await hands.send({ image: videoEl });
            await pose.send({ image: videoEl });

            // draw video + landmarks onto canvas
            canvasEl.width  = videoEl.videoWidth;
            canvasEl.height = videoEl.videoHeight;
            ctx.save();
            // mirror horizontally so it feels like a selfie
            ctx.scale(-1, 1);
            ctx.drawImage(videoEl, -canvasEl.width, 0);
            ctx.restore();

            if (latestHandResult) drawHandLandmarks(ctx, latestHandResult, canvasEl.width);
            if (latestPoseResult) drawPoseLandmarks(ctx, latestPoseResult, canvasEl.width);

            const frameLandmarks = extractLandmarks(latestHandResult, latestPoseResult);
            processFrame(frameLandmarks, latestHandResult, onStateChange);
        },
        width:  1280,
        height: 720
    });

    await camera.start();
    console.log("[inference] Camera loop started");
}

// ── Landmark extraction ───────────────────────────────────────────────────────
// left_hand (63) + right_hand (63) + pose (99) = 225 — mirrors app.py exactly

function extractLandmarks(handResult, poseResult) {
    const leftHand  = new Float32Array(63);
    const rightHand = new Float32Array(63);

    if (handResult?.multiHandLandmarks && handResult?.multiHandedness) {
        handResult.multiHandLandmarks.forEach((landmarks, i) => {
            const label = handResult.multiHandedness[i].label;
            const flat  = landmarksToFlat(landmarks);
            if (label === "Left")  leftHand.set(flat);
            else                   rightHand.set(flat);
        });
    }

    const poseLandmarks = new Float32Array(99);
    if (poseResult?.poseLandmarks) {
        poseLandmarks.set(landmarksToFlat(poseResult.poseLandmarks));
    }

    const frame = new Float32Array(225);
    frame.set(leftHand,      0);
    frame.set(rightHand,    63);
    frame.set(poseLandmarks, 126);
    return frame;
}

function landmarksToFlat(landmarks) {
    const out = new Float32Array(landmarks.length * 3);
    landmarks.forEach((lm, i) => {
        out[i * 3]     = lm.x;
        out[i * 3 + 1] = lm.y;
        out[i * 3 + 2] = lm.z;
    });
    return out;
}

// ── State machine ─────────────────────────────────────────────────────────────
// mirrors generate_frames() in app.py

function processFrame(frameLandmarks, handResult, onStateChange) {
    const handDetected = !!(handResult?.multiHandLandmarks?.length);

    if (smState === "ready") {
        if (handDetected) {
            smState = "signing";
            noHandFrames = 0;
            sequence = [frameLandmarks];
            setState({ state: "signing", sign: "", confidence: 0 }, onStateChange);
        }

    } else if (smState === "signing") {
        sequence.push(frameLandmarks);
        if (sequence.length > SEQUENCE_LENGTH) sequence.shift();

        if (handDetected) {
            noHandFrames = 0;
        } else {
            noHandFrames++;
            if (noHandFrames >= NO_HAND_TRIGGER) {
                smState = "predicting";
                runInference([...sequence]).then(({ sign, confidence }) => {
                    if (sign) {
                        const history = [...predictionState.history];
                        if (!history.length || history[history.length - 1] !== sign) {
                            history.push(sign);
                            if (history.length > 5) history.shift();
                        }
                        setState({ state: "predicting", sign, confidence, history }, onStateChange);
                    } else {
                        setState({ state: "predicting", sign: "", confidence: 0 }, onStateChange);
                    }

                    clearTimeout(predictTimer);
                    predictTimer = setTimeout(() => {
                        smState = "ready";
                        sequence = [];
                        setState({ state: "ready", sign: "", confidence: 0 }, onStateChange);
                    }, PREDICT_RESET_MS);
                });

                sequence = [];
            }
        }

    } else if (smState === "predicting") {
        if (handDetected) {
            clearTimeout(predictTimer);
            smState = "ready";
            sequence = [];
            setState({ state: "ready", sign: "", confidence: 0 }, onStateChange);
        }
    }
}

// ── ONNX inference ────────────────────────────────────────────────────────────
// mirrors run_inference() in app.py

async function runInference(frames) {
    if (frames.length < SEQUENCE_LENGTH) {
        const pad = Array(SEQUENCE_LENGTH - frames.length)
            .fill(null).map(() => new Float32Array(225));
        frames = [...pad, ...frames];
    }

    const flat = new Float32Array(SEQUENCE_LENGTH * 225);
    frames.forEach((frame, i) => flat.set(frame, i * 225));

    const tensor  = new ort.Tensor("float32", flat, [1, SEQUENCE_LENGTH, 225]);
    const results = await onnxSession.run({ input: tensor });
    const logits  = results.output.data;

    // softmax
    const maxLogit = Math.max(...logits);
    const exps     = Array.from(logits).map(v => Math.exp(v - maxLogit));
    const sumExps  = exps.reduce((a, b) => a + b, 0);
    const probs    = exps.map(v => v / sumExps);

    const predicted  = probs.indexOf(Math.max(...probs));
    const confidence = Math.round(probs[predicted] * 100);
    const sign       = SIGNS[predicted];

    if (sign === "nothing") return { sign: null, confidence };
    return { sign, confidence };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setState(updates, onStateChange) {
    predictionState = { ...predictionState, ...updates };
    onStateChange(predictionState);
}

function drawHandLandmarks(ctx, handResult, width) {
    if (!handResult?.multiHandLandmarks) return;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-width, 0);
    handResult.multiHandLandmarks.forEach(landmarks => {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 1 });
        drawLandmarks(ctx, landmarks, { color: "#FF0000", lineWidth: 1, radius: 2 });
    });
    ctx.restore();
}

function drawPoseLandmarks(ctx, poseResult, width) {
    if (!poseResult?.poseLandmarks) return;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-width, 0);
    drawConnectors(ctx, poseResult.poseLandmarks, POSE_CONNECTIONS, { color: "#00FFFF", lineWidth: 1 });
    drawLandmarks(ctx, poseResult.poseLandmarks, { color: "#FF00FF", lineWidth: 1, radius: 2 });
    ctx.restore();
}