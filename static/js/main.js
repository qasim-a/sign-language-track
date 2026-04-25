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