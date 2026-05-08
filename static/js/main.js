// --- Normal mode elements ---
const signEl = document.getElementById('sign');
const confidenceText = document.getElementById('confidence-text');
const confidenceBar = document.getElementById('confidence-bar');
const historyEl = document.getElementById('history');

let lastAddedSign = '';
let signHeldCount = 0;
const HOLD_THRESHOLD = 2;

function updatePrediction() {
    fetch('/prediction')
        .then(res => res.json())
        .then(data => {
            if (data.sign) {
                signEl.textContent = data.sign.charAt(0).toUpperCase() + data.sign.slice(1);
                confidenceText.textContent = '';
                const normalizedConfidence = Math.min(((data.confidence - 90) / 10) * 100, 100);
                confidenceBar.style.width = `${normalizedConfidence}%`;

                if (data.sign === lastAddedSign) {
                    signHeldCount++;
                } else {
                    signHeldCount = 1;
                    lastAddedSign = data.sign;
                }

                if (signHeldCount === HOLD_THRESHOLD) {
                    addToHistory(data.sign);
                }
            } else {
                signEl.textContent = '';
                confidenceText.textContent = '';
                confidenceBar.style.width = '0%';
                lastAddedSign = '';
                signHeldCount = 0;
            }
        });
}

function addToHistory(sign) {
    const items = historyEl.querySelectorAll('.history-item');
    if (items.length > 0 && items[0].dataset.sign === sign) return;

    const item = document.createElement('div');
    item.className = 'history-item';
    item.dataset.sign = sign;
    item.textContent = sign.charAt(0).toUpperCase() + sign.slice(1);

    historyEl.insertBefore(item, historyEl.firstChild);

    while (historyEl.children.length > 5) {
        historyEl.removeChild(historyEl.lastChild);
    }

    const noSigns = historyEl.querySelector('.no-signs');
    if (noSigns) noSigns.remove();
}

historyEl.innerHTML = '<div class="history-item no-signs" style="color: #bbb">No signs detected yet</div>';

setInterval(updatePrediction, 300);

// --- Test Mode ---

const TEST_SIGNS = ["hello", "yes", "no", "thank you", "please", "eat", "drink", "water", "more", "apple", "mother"];
const ATTEMPT_DURATION_MS = 8000;
const ATTEMPTS_PER_SIGN = 3;
const TEST_CONFIDENCE_THRESHOLD = 70; // lower than live mode to capture more data

let testRunning = false;
let testResults = [];
let attemptTimer = null;
let attemptStartTime = null;
let timerAnimFrame = null;
let attemptResolved = false;
let pollInterval = null;

// tracks consecutive polls of same sign — mirrors recent signs hold logic
let testLastSeen = '';
let testHeldCount = 0;

const testOverlay = document.getElementById('test-overlay');
const resultsOverlay = document.getElementById('results-overlay');
const startTestBtn = document.getElementById('start-test-btn');
const cancelTestBtn = document.getElementById('cancel-test-btn');
const copyResultsBtn = document.getElementById('copy-results-btn');
const closeResultsBtn = document.getElementById('close-results-btn');

startTestBtn.addEventListener('click', startTest);
cancelTestBtn.addEventListener('click', cancelTest);
copyResultsBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('results-text').textContent);
    copyResultsBtn.textContent = 'Copied';
    setTimeout(() => copyResultsBtn.textContent = 'Copy Results', 1500);
});
closeResultsBtn.addEventListener('click', () => {
    resultsOverlay.classList.remove('active');
});

function startTest() {
    testRunning = true;
    testResults = [];
    testOverlay.classList.add('active');
    loadSign(0);
}

function cancelTest() {
    testRunning = false;
    testOverlay.classList.remove('active');
    clearTimeout(attemptTimer);
    cancelAnimationFrame(timerAnimFrame);
    clearInterval(pollInterval);
}

function loadSign(signIdx) {
    testResults[signIdx] = { sign: TEST_SIGNS[signIdx], attempts: [] };

    document.getElementById('test-progress').textContent = `Sign ${signIdx + 1} of ${TEST_SIGNS.length}`;
    document.getElementById('test-sign-name').textContent = capitalize(TEST_SIGNS[signIdx]);

    resetDots();
    startAttempt(signIdx, 0);
}

function resetDots() {
    for (let i = 0; i < ATTEMPTS_PER_SIGN; i++) {
        const dot = document.getElementById(`dot-${i}`);
        dot.className = 'attempt-dot';
        dot.textContent = i + 1;
    }
    document.getElementById('dot-0').classList.add('active');
}

function startAttempt(signIdx, attemptIdx) {
    attemptResolved = false;
    attemptStartTime = Date.now();
    testLastSeen = '';
    testHeldCount = 0;

    document.getElementById(`dot-${attemptIdx}`).classList.add('active');
    document.getElementById('test-live-result').textContent = 'Waiting for sign...';
    document.getElementById('test-live-result').className = 'test-live-result';
    document.getElementById('test-status-msg').textContent = `Attempt ${attemptIdx + 1} of ${ATTEMPTS_PER_SIGN} — sign once, hold briefly`;

    animateTimer();

    // poll every 300ms — two consecutive polls of same sign resolves the attempt
    pollInterval = setInterval(() => {
        if (attemptResolved) return;
        fetch('/prediction')
            .then(res => res.json())
            .then(data => {
                if (attemptResolved) return;
                const liveEl = document.getElementById('test-live-result');
                const expected = TEST_SIGNS[signIdx].toLowerCase();

                // use lower confidence threshold for test mode
                const confident = data.confidence >= TEST_CONFIDENCE_THRESHOLD;
                const detected = confident && data.sign ? data.sign.toLowerCase() : null;

                if (detected) {
                    if (detected === testLastSeen) {
                        testHeldCount++;
                    } else {
                        testLastSeen = detected;
                        testHeldCount = 1;
                    }

                    if (detected === expected) {
                        liveEl.textContent = `${capitalize(detected)} (${data.confidence}%)`;
                        liveEl.className = 'test-live-result correct';
                    } else {
                        liveEl.textContent = `${capitalize(detected)} (${data.confidence}%)`;
                        liveEl.className = 'test-live-result incorrect';
                    }

                    // two consecutive polls of same sign = resolve attempt immediately
                    if (testHeldCount >= 2) {
                        clearInterval(pollInterval);
                        clearTimeout(attemptTimer);
                        cancelAnimationFrame(timerAnimFrame);
                        attemptResolved = true;
                        recordAttempt(signIdx, attemptIdx, detected, data.confidence);
                    }
                } else {
                    testLastSeen = '';
                    testHeldCount = 0;
                    // show low confidence detection if present, helps diagnose blanks
                    if (data.sign && !confident) {
                        liveEl.textContent = `${capitalize(data.sign)} (${data.confidence}% — below threshold)`;
                    } else {
                        liveEl.textContent = 'Waiting for sign...';
                    }
                    liveEl.className = 'test-live-result';
                }
            });
    }, 300);

    // 5 second max — records blank if nothing held for 2 consecutive polls
    attemptTimer = setTimeout(() => {
        if (attemptResolved) return;
        clearInterval(pollInterval);
        cancelAnimationFrame(timerAnimFrame);
        attemptResolved = true;
        if (testLastSeen) {
            recordAttempt(signIdx, attemptIdx, testLastSeen, null);
        } else {
            recordAttempt(signIdx, attemptIdx, null, null);
        }
    }, ATTEMPT_DURATION_MS);
}

function recordAttempt(signIdx, attemptIdx, detectedSign, confidence) {
    const expected = TEST_SIGNS[signIdx].toLowerCase();
    const detected = detectedSign ? detectedSign.toLowerCase() : null;

    let result;
    if (!detected) {
        result = 'blank';
    } else if (detected === expected) {
        result = 'correct';
    } else {
        result = 'incorrect';
    }

    testResults[signIdx].attempts.push({ result, detected, confidence });

    const dot = document.getElementById(`dot-${attemptIdx}`);
    dot.classList.remove('active');
    if (result === 'correct') { dot.classList.add('correct'); dot.textContent = 'OK'; }
    else if (result === 'incorrect') { dot.classList.add('incorrect'); dot.textContent = 'X'; }
    else { dot.classList.add('blank'); dot.textContent = '-'; }

    const nextAttempt = attemptIdx + 1;

    if (nextAttempt < ATTEMPTS_PER_SIGN) {
        document.getElementById('test-status-msg').textContent = 'Drop your hand — next attempt in 2 seconds';
        setTimeout(() => {
            if (!testRunning) return;
            startAttempt(signIdx, nextAttempt);
        }, 2000);
    } else {
        const nextSign = signIdx + 1;
        if (nextSign < TEST_SIGNS.length) {
            document.getElementById('test-status-msg').textContent = 'Next sign in 2 seconds...';
            setTimeout(() => {
                if (!testRunning) return;
                loadSign(nextSign);
            }, 2000);
        } else {
            finishTest();
        }
    }
}

function animateTimer() {
    cancelAnimationFrame(timerAnimFrame);
    const bar = document.getElementById('test-timer-bar');

    function tick() {
        const elapsed = Date.now() - attemptStartTime;
        const pct = Math.max(0, 100 - (elapsed / ATTEMPT_DURATION_MS) * 100);
        bar.style.width = `${pct}%`;
        if (pct > 0 && !attemptResolved) {
            timerAnimFrame = requestAnimationFrame(tick);
        }
    }
    timerAnimFrame = requestAnimationFrame(tick);
}

function finishTest() {
    testRunning = false;
    testOverlay.classList.remove('active');
    cancelAnimationFrame(timerAnimFrame);

    const date = new Date().toLocaleString();
    let totalAttempts = 0;
    let totalCorrect = 0;
    let lines = [];

    lines.push(`SignLanguageTrack Test Results`);
    lines.push(`${date}`);
    lines.push(`Signs tested: ${TEST_SIGNS.length}`);
    lines.push(``);

    testResults.forEach(signResult => {
        const name = capitalize(signResult.sign);
        lines.push(`${name}`);
        signResult.attempts.forEach((a, i) => {
            totalAttempts++;
            if (a.result === 'correct') {
                totalCorrect++;
                const conf = a.confidence ? ` (${a.confidence}%)` : '';
                lines.push(`  Attempt ${i + 1}: correct${conf}`);
            } else if (a.result === 'incorrect') {
                const conf = a.confidence ? ` (${a.confidence}%)` : '';
                lines.push(`  Attempt ${i + 1}: incorrect — recognized as ${capitalize(a.detected)}${conf}`);
            } else {
                lines.push(`  Attempt ${i + 1}: blank — nothing recognized`);
            }
        });
        lines.push(``);
    });

    const pct = Math.round((totalCorrect / totalAttempts) * 100);
    lines.push(`Overall: ${totalCorrect}/${totalAttempts} correct (${pct}%)`);

    document.getElementById('results-text').textContent = lines.join('\n');
    document.getElementById('results-summary').textContent =
        `${totalCorrect} of ${totalAttempts} attempts correct (${pct}%)`;

    resultsOverlay.classList.add('active');
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}