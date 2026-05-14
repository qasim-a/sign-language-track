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

const ALL_SIGNS = ["hello", "yes", "no", "thank you", "please", "eat", "drink", "water", "more", "apple", "mother", "father", "book"];
const ATTEMPTS_ALL    = 6;   // attempts per sign when testing all signs
const ATTEMPTS_CUSTOM = 9;   // attempts per sign when testing a custom selection
const RESULT_DISPLAY_MS = 1200;

let testRunning    = false;
let testSigns      = [];     // signs selected for this run
let attemptsPerSign = ATTEMPTS_ALL;
let testResults    = [];
let pollInterval   = null;
let selectedSigns  = new Set();  // tracks which signs are toggled in the picker

const selectionOverlay = document.getElementById('selection-overlay');
const testOverlay      = document.getElementById('test-overlay');
const resultsOverlay   = document.getElementById('results-overlay');
const startTestBtn     = document.getElementById('start-test-btn');
const cancelTestBtn    = document.getElementById('cancel-test-btn');
const copyResultsBtn   = document.getElementById('copy-results-btn');
const closeResultsBtn  = document.getElementById('close-results-btn');
const testAllBtn       = document.getElementById('test-all-btn');
const startCustomBtn   = document.getElementById('start-custom-btn');
const cancelSelBtn     = document.getElementById('cancel-sel-btn');

// ── Selection screen ──────────────────────────────────────────────────────────

startTestBtn.addEventListener('click', () => {
    selectedSigns.clear();
    updateSignGrid();
    updateCustomBtn();
    selectionOverlay.classList.add('active');
});

cancelSelBtn.addEventListener('click', () => {
    selectionOverlay.classList.remove('active');
});

testAllBtn.addEventListener('click', () => {
    selectionOverlay.classList.remove('active');
    beginTest(ALL_SIGNS, ATTEMPTS_ALL);
});

startCustomBtn.addEventListener('click', () => {
    if (selectedSigns.size === 0) return;
    selectionOverlay.classList.remove('active');
    // preserve the original sign order
    const ordered = ALL_SIGNS.filter(s => selectedSigns.has(s));
    beginTest(ordered, ATTEMPTS_CUSTOM);
});

function updateSignGrid() {
    const grid = document.getElementById('sign-grid');
    grid.innerHTML = '';
    ALL_SIGNS.forEach(sign => {
        const tag = document.createElement('button');
        tag.className = 'sign-tag' + (selectedSigns.has(sign) ? ' selected' : '');
        tag.textContent = capitalize(sign);
        tag.addEventListener('click', () => {
            if (selectedSigns.has(sign)) {
                selectedSigns.delete(sign);
                tag.classList.remove('selected');
            } else {
                selectedSigns.add(sign);
                tag.classList.add('selected');
            }
            updateCustomBtn();
        });
        grid.appendChild(tag);
    });
}

function updateCustomBtn() {
    const n = selectedSigns.size;
    if (n === 0) {
        startCustomBtn.textContent = 'Select signs to start';
        startCustomBtn.disabled = true;
    } else {
        startCustomBtn.textContent = `Start Custom Test · ${n} sign${n > 1 ? 's' : ''} · 9 attempts each`;
        startCustomBtn.disabled = false;
    }
}

// ── Test runner ───────────────────────────────────────────────────────────────

function beginTest(signs, attempts) {
    testSigns       = signs;
    attemptsPerSign = attempts;
    testRunning     = true;
    testResults     = signs.map(sign => ({ sign, attempts: [] }));

    // build dots dynamically based on attempt count
    buildDots(attempts);

    testOverlay.classList.add('active');
    runAttempt(0, 0);
}

function buildDots(count) {
    const container = document.getElementById('test-attempts');
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const dot = document.createElement('div');
        dot.className = 'attempt-dot' + (i === 0 ? ' active' : '');
        dot.id = `dot-${i}`;
        dot.textContent = i + 1;
        container.appendChild(dot);
    }
}

cancelTestBtn.addEventListener('click', cancelTest);

copyResultsBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('results-text').textContent);
    copyResultsBtn.textContent = 'Copied';
    setTimeout(() => copyResultsBtn.textContent = 'Copy Results', 1500);
});

closeResultsBtn.addEventListener('click', () => resultsOverlay.classList.remove('active'));

function cancelTest() {
    testRunning = false;
    clearInterval(pollInterval);
    testOverlay.classList.remove('active');
}

// ── Core loop ─────────────────────────────────────────────────────────────────

function runAttempt(signIdx, attemptIdx) {
    if (!testRunning) return;

    // silent delay before the very first attempt only
    if (signIdx === 0 && attemptIdx === 0) {
        setTimeout(() => runAttempt_inner(signIdx, attemptIdx), 1200);
        return;
    }
    runAttempt_inner(signIdx, attemptIdx);
}

function runAttempt_inner(signIdx, attemptIdx) {
    if (!testRunning) return;

    const sign = testSigns[signIdx];

    document.getElementById('test-progress').textContent =
        `Sign ${signIdx + 1} of ${testSigns.length}`;
    document.getElementById('test-sign-name').textContent = capitalize(sign);
    document.getElementById('test-attempt-count').textContent =
        `Attempt ${attemptIdx + 1} of ${attemptsPerSign}`;
    updateDots(signIdx, attemptIdx, null);

    setStatus('ready-wait', '');
    clearInterval(pollInterval);

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

    // if model returned nothing, silently retry the same attempt
    if (!detectedSign) {
        runAttempt(signIdx, attemptIdx);
        return;
    }

    const sign      = testSigns[signIdx];
    const detected  = detectedSign.toLowerCase();
    const isCorrect = detected === sign.toLowerCase();

    testResults[signIdx].attempts.push({ detected: detectedSign, confidence: confidence || 0 });
    updateDots(signIdx, attemptIdx, isCorrect ? 'correct' : 'incorrect');
    setStatus('result', `${capitalize(detectedSign)} — ${confidence}%`, isCorrect ? 'correct' : 'incorrect');

    setTimeout(() => {
        if (!testRunning) return;
        const nextAttempt = attemptIdx + 1;
        if (nextAttempt < attemptsPerSign) {
            runAttempt(signIdx, nextAttempt);
        } else {
            const nextSign = signIdx + 1;
            if (nextSign < testSigns.length) {
                runAttempt(nextSign, 0);
            } else {
                finishTest();
            }
        }
    }, RESULT_DISPLAY_MS);
}

// ── Dot helpers ───────────────────────────────────────────────────────────────

function updateDots(signIdx, currentAttempt, currentResult) {
    for (let i = 0; i < attemptsPerSign; i++) {
        const dot = document.getElementById(`dot-${i}`);
        if (!dot) continue;
        dot.className = 'attempt-dot';
        dot.textContent = i + 1;

        if (i < currentAttempt) {
            const a = testResults[signIdx].attempts[i];
            if (a) {
                const correct = a.detected && a.detected.toLowerCase() === testSigns[signIdx].toLowerCase();
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
    const liveEl   = document.getElementById('test-live-result');
    const statusEl = document.getElementById('test-status-msg');

    liveEl.className = 'test-live-result';
    statusEl.className = 'test-status-msg';

    if (type === 'ready-wait') {
        liveEl.textContent   = '';
        statusEl.textContent = '';
    } else if (type === 'ready') {
        liveEl.textContent   = 'Ready';
        statusEl.textContent = "Sign when you're set";
    } else if (type === 'signing') {
        liveEl.textContent   = 'Signing detected…';
        statusEl.textContent = 'Drop your hand when done';
    } else if (type === 'result') {
        liveEl.textContent   = message;
        liveEl.classList.add(tone === 'correct' ? 'correct' : tone === 'incorrect' ? 'incorrect' : '');
        statusEl.textContent = '';
    }
}

// ── Results ───────────────────────────────────────────────────────────────────

function finishTest() {
    testRunning = false;
    clearInterval(pollInterval);
    testOverlay.classList.remove('active');

    const date     = new Date().toLocaleString();
    const mode     = testSigns.length === ALL_SIGNS.length ? 'All Signs' : 'Custom Selection';
    let totalAttempts = 0;
    let totalCorrect  = 0;
    let lines = [];

    lines.push(`SignLanguageTrack Test Results`);
    lines.push(`${date}`);
    lines.push(`Mode: ${mode}  ·  Signs tested: ${testSigns.length}  ·  Attempts per sign: ${attemptsPerSign}`);
    lines.push(``);

    testResults.forEach(signResult => {
        lines.push(`${capitalize(signResult.sign)}`);
        signResult.attempts.forEach((a, i) => {
            totalAttempts++;
            const correct = a.detected && a.detected.toLowerCase() === signResult.sign.toLowerCase();
            if (correct) totalCorrect++;
            const conf = a.confidence ? `${a.confidence}%` : '—';
            if (correct) {
                lines.push(`  Attempt ${i + 1}: correct · ${conf}`);
            } else {
                const det = a.detected ? capitalize(a.detected) : '—';
                lines.push(`  Attempt ${i + 1}: incorrect · recognized as ${det} · ${conf}`);
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