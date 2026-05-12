// --- Normal mode elements ---
const signEl = document.getElementById('sign');
const confidenceText = document.getElementById('confidence-text');
const confidenceBar = document.getElementById('confidence-bar');
const historyEl = document.getElementById('history');
const stateIndicator = document.getElementById('state-indicator');

const STATE_CONFIG = {
    ready:      { label: 'Ready',    cls: 'state-ready' },
    signing:    { label: 'Signing…', cls: 'state-signing' },
    predicting: { label: 'Done',     cls: 'state-predicting' },
};

let lastState = '';

function updatePrediction() {
    fetch('/prediction')
        .then(res => res.json())
        .then(data => {
            const state = data.state || 'ready';

            if (state !== lastState) {
                const cfg = STATE_CONFIG[state] || STATE_CONFIG.ready;
                stateIndicator.textContent = cfg.label;
                stateIndicator.className = 'state-indicator ' + cfg.cls;
                lastState = state;
            }

            if (state === 'predicting') {
                if (data.sign) {
                    signEl.textContent = capitalize(data.sign);
                    confidenceText.textContent = data.confidence + '%';
                    confidenceBar.style.width = `${data.confidence}%`;
                    addToHistory(data.sign);
                } else {
                    signEl.textContent = '—';
                    confidenceText.textContent = '';
                    confidenceBar.style.width = '0%';
                }
            } else if (state === 'ready') {
                signEl.textContent = '—';
                confidenceText.textContent = '';
                confidenceBar.style.width = '0%';
            }
        });
}

function addToHistory(sign) {
    const items = historyEl.querySelectorAll('.history-item');
    if (items.length > 0 && items[0].dataset.sign === sign) return;

    const noSigns = historyEl.querySelector('.no-signs');
    if (noSigns) noSigns.remove();

    const item = document.createElement('div');
    item.className = 'history-item';
    item.dataset.sign = sign;
    item.textContent = capitalize(sign);
    historyEl.insertBefore(item, historyEl.firstChild);

    while (historyEl.children.length > 5) {
        historyEl.removeChild(historyEl.lastChild);
    }
}

historyEl.innerHTML = '<div class="history-item no-signs" style="color: #bbb">No signs detected yet</div>';

setInterval(updatePrediction, 150);

// ─── Test Mode ────────────────────────────────────────────────────────────────

const TEST_SIGNS = ["hello", "yes", "no", "thank you", "please", "eat", "drink", "water", "more", "apple", "mother"];
const ATTEMPTS_PER_SIGN = 3;
const RESULT_DISPLAY_MS = 1200;   // how long to show each result before moving on

let testRunning = false;
let testResults = [];   // [{ sign, attempts: [{ detected, confidence }] }]
let pollInterval = null;

const testOverlay    = document.getElementById('test-overlay');
const resultsOverlay = document.getElementById('results-overlay');
const startTestBtn   = document.getElementById('start-test-btn');
const cancelTestBtn  = document.getElementById('cancel-test-btn');
const copyResultsBtn = document.getElementById('copy-results-btn');
const closeResultsBtn = document.getElementById('close-results-btn');

startTestBtn.addEventListener('click', startTest);
cancelTestBtn.addEventListener('click', cancelTest);
copyResultsBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('results-text').textContent);
    copyResultsBtn.textContent = 'Copied';
    setTimeout(() => copyResultsBtn.textContent = 'Copy Results', 1500);
});
closeResultsBtn.addEventListener('click', () => resultsOverlay.classList.remove('active'));

function startTest() {
    testRunning = true;
    testResults = TEST_SIGNS.map(sign => ({ sign, attempts: [] }));
    testOverlay.classList.add('active');
    runAttempt(0, 0);
}

function cancelTest() {
    testRunning = false;
    clearInterval(pollInterval);
    testOverlay.classList.remove('active');
}

// ── Core loop: wait for one full ready→signing→predicting cycle ───────────────
function runAttempt(signIdx, attemptIdx) {
    if (!testRunning) return;

    const sign = TEST_SIGNS[signIdx];

    // update UI for this attempt
    document.getElementById('test-progress').textContent =
        `Sign ${signIdx + 1} of ${TEST_SIGNS.length}`;
    document.getElementById('test-sign-name').textContent = capitalize(sign);
    document.getElementById('test-attempt-count').textContent =
        `Attempt ${attemptIdx + 1} of ${ATTEMPTS_PER_SIGN}`;
    updateDots(signIdx, attemptIdx, null);   // mark current dot as active, rest neutral

    setStatus('ready-wait', '');    // waiting for backend ready state
    clearInterval(pollInterval);

    // phase 1 — wait for 'ready' so we don't catch a stale result
    let phase = 'waitingForReady';

    pollInterval = setInterval(() => {
        if (!testRunning) return;
        fetch('/prediction')
            .then(res => res.json())
            .then(data => {
                if (!testRunning) return;
                const state = data.state || 'ready';

                if (phase === 'waitingForReady') {
                    if (state === 'ready') {
                        phase = 'waitingForSign';
                        setStatus('ready', '');
                    }
                    return;
                }

                if (phase === 'waitingForSign') {
                    if (state === 'signing') {
                        phase = 'signing';
                        setStatus('signing', '');
                    }
                    return;
                }

                if (phase === 'signing') {
                    if (state === 'predicting') {
                        phase = 'done';
                        clearInterval(pollInterval);
                        handleResult(signIdx, attemptIdx, data.sign, data.confidence);
                    }
                    // if state goes back to ready without signing→predicting
                    // (e.g. very brief hand flash), restart the wait
                    if (state === 'ready') {
                        phase = 'waitingForSign';
                        setStatus('ready', '');
                    }
                    return;
                }
            });
    }, 150);
}

function handleResult(signIdx, attemptIdx, detectedSign, confidence) {
    if (!testRunning) return;

    const sign = TEST_SIGNS[signIdx];
    const detected = detectedSign || null;
    const isCorrect = detected && detected.toLowerCase() === sign.toLowerCase();

    // record
    testResults[signIdx].attempts.push({ detected, confidence: confidence || 0 });

    // update dot
    updateDots(signIdx, attemptIdx, isCorrect ? 'correct' : 'incorrect');

    // show result briefly
    if (detected) {
        setStatus('result', `${capitalize(detected)} — ${confidence}%`, isCorrect ? 'correct' : 'incorrect');
    } else {
        setStatus('result', 'No sign detected', 'neutral');
    }

    // advance after short display pause
    setTimeout(() => {
        if (!testRunning) return;

        const nextAttempt = attemptIdx + 1;
        if (nextAttempt < ATTEMPTS_PER_SIGN) {
            runAttempt(signIdx, nextAttempt);
        } else {
            const nextSign = signIdx + 1;
            if (nextSign < TEST_SIGNS.length) {
                runAttempt(nextSign, 0);
            } else {
                finishTest();
            }
        }
    }, RESULT_DISPLAY_MS);
}

// ── Dot state helpers ─────────────────────────────────────────────────────────
// currentResult: null (active/pending) | 'correct' | 'incorrect'
function updateDots(signIdx, currentAttempt, currentResult) {
    for (let i = 0; i < ATTEMPTS_PER_SIGN; i++) {
        const dot = document.getElementById(`dot-${i}`);
        dot.className = 'attempt-dot';
        dot.textContent = i + 1;

        if (i < currentAttempt) {
            // already recorded — look up stored result
            const a = testResults[signIdx].attempts[i];
            if (a) {
                const correct = a.detected && a.detected.toLowerCase() === TEST_SIGNS[signIdx].toLowerCase();
                dot.classList.add(correct ? 'correct' : 'incorrect');
                dot.textContent = correct ? '✓' : '✗';
            }
        } else if (i === currentAttempt) {
            if (currentResult === 'correct')        { dot.classList.add('correct');   dot.textContent = '✓'; }
            else if (currentResult === 'incorrect') { dot.classList.add('incorrect'); dot.textContent = '✗'; }
            else                                    { dot.classList.add('active'); }
        }
    }
}

// ── Status display ────────────────────────────────────────────────────────────
function setStatus(type, message, tone) {
    const liveEl  = document.getElementById('test-live-result');
    const statusEl = document.getElementById('test-status-msg');

    liveEl.className = 'test-live-result';
    statusEl.className = 'test-status-msg';

    if (type === 'ready-wait') {
        liveEl.textContent = '';
        statusEl.textContent = 'Waiting for ready state…';
    } else if (type === 'ready') {
        liveEl.textContent = '';
        statusEl.textContent = 'Ready — sign when you\'re set';
    } else if (type === 'signing') {
        liveEl.textContent = 'Signing detected…';
        statusEl.textContent = 'Drop your hand when done';
    } else if (type === 'result') {
        liveEl.textContent = message;
        liveEl.classList.add(tone === 'correct' ? 'correct' : tone === 'incorrect' ? 'incorrect' : '');
        statusEl.textContent = '';
    }
}

// ── Results ───────────────────────────────────────────────────────────────────
function finishTest() {
    testRunning = false;
    clearInterval(pollInterval);
    testOverlay.classList.remove('active');

    const date = new Date().toLocaleString();
    let totalAttempts = 0;
    let totalCorrect  = 0;
    let lines = [];

    lines.push(`SignLanguageTrack Test Results`);
    lines.push(`${date}`);
    lines.push(`Signs tested: ${TEST_SIGNS.length}  ·  Attempts per sign: ${ATTEMPTS_PER_SIGN}`);
    lines.push(``);

    testResults.forEach(signResult => {
        lines.push(`${capitalize(signResult.sign)}`);
        signResult.attempts.forEach((a, i) => {
            totalAttempts++;
            const correct = a.detected && a.detected.toLowerCase() === signResult.sign.toLowerCase();
            if (correct) totalCorrect++;
            const label  = correct ? 'correct' : 'incorrect';
            const det    = a.detected ? capitalize(a.detected) : '—';
            const conf   = a.confidence ? `${a.confidence}%` : '—';
            if (correct) {
                lines.push(`  Attempt ${i + 1}: ${label} · ${conf}`);
            } else {
                lines.push(`  Attempt ${i + 1}: ${label} · recognized as ${det} · ${conf}`);
            }
        });
        lines.push(``);
    });

    const pct = Math.round((totalCorrect / totalAttempts) * 100);
    lines.push(`Overall: ${totalCorrect}/${totalAttempts} correct (${pct}%)`);

    document.getElementById('results-text').textContent = lines.join('\n');
    document.getElementById('results-summary').textContent =
        `${totalCorrect} of ${totalAttempts} attempts correct · ${pct}%`;

    resultsOverlay.classList.add('active');
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}