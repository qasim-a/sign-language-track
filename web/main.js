import { startInference } from './inference.js';

// --- Normal mode elements ---
const signEl         = document.getElementById('sign');
const confidenceText = document.getElementById('confidence-text');
const confidenceBar  = document.getElementById('confidence-bar');
const historyEl      = document.getElementById('history');
const stateIndicator = document.getElementById('state-indicator');
const statusLabel    = document.getElementById('status-label');
const loadingMsg     = document.getElementById('loading-msg');
const videoEl        = document.getElementById('webcam');
const canvasEl       = document.getElementById('overlay');

// --- How to Use modal ---
const howToOverlay = document.getElementById('how-to-overlay');
const howToBtn     = document.getElementById('how-to-btn');
const gotItBtn     = document.getElementById('got-it-btn');

// show on load (overlay starts with .active class in HTML)
// dismiss on Got it
gotItBtn.addEventListener('click', () => {
    howToOverlay.classList.remove('active');
});

// reopen from header button
howToBtn.addEventListener('click', () => {
    howToOverlay.classList.add('active');
});

// also dismiss by clicking the dark backdrop
howToOverlay.addEventListener('click', e => {
    if (e.target === howToOverlay) howToOverlay.classList.remove('active');
});

// --- State machine config ---
const STATE_CONFIG = {
    ready:      { label: 'Ready',    cls: 'state-ready' },
    signing:    { label: 'Signing…', cls: 'state-signing' },
    predicting: { label: 'Done',     cls: 'state-predicting' },
};

let lastState = '';

// called by inference.js whenever state changes
function onStateChange(data) {
    const state = data.state || 'ready';

    if (state !== lastState) {
        const cfg = STATE_CONFIG[state] || STATE_CONFIG.ready;
        stateIndicator.textContent = cfg.label;
        stateIndicator.className   = 'state-indicator ' + cfg.cls;
        lastState = state;
    }

    if (state === 'predicting') {
        if (data.sign) {
            signEl.textContent         = capitalize(data.sign);
            confidenceText.textContent = data.confidence + '%';
            confidenceBar.style.width  = `${data.confidence}%`;
            addToHistory(data.sign);
        } else {
            signEl.textContent         = '—';
            confidenceText.textContent = '';
            confidenceBar.style.width  = '0%';
        }
    } else if (state === 'ready') {
        signEl.textContent         = '—';
        confidenceText.textContent = '';
        confidenceBar.style.width  = '0%';
    }
}

function addToHistory(sign) {
    const items = historyEl.querySelectorAll('.history-item');
    if (items.length > 0 && items[0].dataset.sign === sign) return;

    const noSigns = historyEl.querySelector('.no-signs');
    if (noSigns) noSigns.remove();

    const item = document.createElement('div');
    item.className    = 'history-item';
    item.dataset.sign = sign;
    item.textContent  = capitalize(sign);
    historyEl.insertBefore(item, historyEl.firstChild);

    while (historyEl.children.length > 5) {
        historyEl.removeChild(historyEl.lastChild);
    }
}

historyEl.innerHTML = '<div class="history-item no-signs" style="color:#bbb">No signs detected yet</div>';

// boot inference engine
startInference(videoEl, canvasEl, onStateChange)
    .then(() => {
        loadingMsg.style.display = 'none';
        statusLabel.textContent  = 'Live';
    })
    .catch(err => {
        loadingMsg.textContent = 'Camera error — please allow access and refresh.';
        console.error('[main]', err);
    });

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}