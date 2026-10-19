/* ─────────────────────────────────────────────────────────────
   Transformer Explainer — app.js
   Full interactive engine for the Polo Club visualization.
───────────────────────────────────────────────────────────── */

const API = (typeof window !== 'undefined' && window.location.origin && window.location.origin !== 'null') 
  ? `${window.location.origin}/api` 
  : 'http://127.0.0.1:8080/api';

// ── Application State ───────────────────────────────────────
let state = {
  text: 'Data visualization empowers users to',
  tokens: ['Data', 'visualization', 'em', 'powers', 'users', 'to'],
  tokenIds: [],
  attentions: [],       // [12][12][seq][seq]
  hiddenStates: [],
  probabilities: [],
  sampled: { token: 'make', probability: 0.0626, logit: -137.65 },
  currentBlock: 0,      // 0 to 11
  currentHead: 0,       // 0 to 11
  temperature: 0.8,
  samplingType: 'top-k',
  samplingValue: 5,
  isThreeStageOpen: false,
  isProbDetailsOpen: false,
  isRunning: false,
};

const EXAMPLES = [
  'Data visualization empowers users to',
  'Artificial Intelligence is transforming the',
  'As the spaceship entered the orbit of',
  'On the deserted planet, the explorer found',
  'IEEE VIS conference highlights the',
];

// ── Lifecycle ───────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initHeatmapSelectors();
  renderAll();
  generatePrediction();

  window.addEventListener('resize', debounce(() => renderRibbons(), 120));
  window.addEventListener('click', (e) => {
    // Hide dropdown when clicking outside
    if (!e.target.closest('.dropdown-wrapper')) {
      const menu = document.getElementById('examples-menu');
      if (menu) menu.classList.add('hidden');
    }
  });
});

// ── Inference ───────────────────────────────────────────────
async function generatePrediction() {
  if (state.isRunning) return;
  state.isRunning = true;
  setLoadingState(true);

  const inputEl = document.getElementById('input-text-display');
  const txt = (inputEl.innerText || inputEl.textContent || state.text).trim() || 'Data visualization empowers users to';
  state.text = txt;

  try {
    const res = await fetch(`${API}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: state.text,
        temperature: state.temperature,
        sampling_type: state.samplingType,
        sampling_value: state.samplingValue,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      state.tokens = data.tokens.map(cleanToken);
      state.tokenIds = data.tokenIds;
      state.attentions = data.attentions;
      state.hiddenStates = data.hiddenStates;
      state.probabilities = data.probabilities;
      state.sampled = data.sampled;
      
      const badge = document.getElementById('predicted-token-badge');
      if (badge && state.sampled) {
        badge.textContent = cleanToken(state.sampled.token);
      }
    } else {
      useFallbackData();
    }
  } catch (err) {
    useFallbackData();
  } finally {
    setLoadingState(false);
    state.isRunning = false;
    renderAll();
  }
}

function cleanToken(tok) {
  if (!tok) return '';
  return tok.replace(/Ġ/g, '').replace(/\[NEWLINE\]/g, '↵').replace(/\[TAB\]/g, '⇥');
}

function useFallbackData() {
  const words = state.text.trim().split(/\s+/);
  state.tokens = words.length > 0 ? words : ['Data', 'visualization', 'em', 'powers', 'users', 'to'];
  const n = state.tokens.length;

  state.attentions = Array.from({ length: 12 }, () =>
    Array.from({ length: 12 }, () =>
      Array.from({ length: n }, (_, r) => {
        const row = Array.from({ length: n }, (_, c) => (c <= r ? Math.random() : 0));
        const sum = row.reduce((a, b) => a + b, 0) || 1;
        return row.map(v => v / sum);
      })
    )
  );

  state.probabilities = [
    { token: 'visualize', probability: 0.5467, logit: -135.91, scaledLogit: -169.89 },
    { token: 'create', probability: 0.2087, logit: -136.68, scaledLogit: -170.85 },
    { token: 'see', probability: 0.1209, logit: -137.12, scaledLogit: -171.40 },
    { token: 'make', probability: 0.0626, logit: -137.65, scaledLogit: -172.06 },
    { token: 'easily', probability: 0.0611, logit: -137.67, scaledLogit: -172.08 },
    { token: 'quickly', probability: 0.0000, logit: -137.72, scaledLogit: -172.15 },
    { token: 'explore', probability: 0.0000, logit: -137.78, scaledLogit: -172.23 },
    { token: 'find', probability: 0.0000, logit: -138.05, scaledLogit: -172.56 },
    { token: 'understand', probability: 0.0000, logit: -138.29, scaledLogit: -172.87 },
    { token: 'build', probability: 0.0000, logit: -138.41, scaledLogit: -173.01 },
  ];
  state.sampled = state.probabilities[3]; // 'make'
}

// ── Rendering ───────────────────────────────────────────────
function renderAll() {
  renderTokens();
  renderQkv();
  renderAttentionCard();
  renderMlp();
  renderSubsequent();
  renderProbabilities();
  requestAnimationFrame(() => renderRibbons());
}

// 1. Tokens
function renderTokens() {
  const container = document.getElementById('col-tokens');
  if (!container) return;
  container.innerHTML = '';

  state.tokens.forEach((tok, i) => {
    const el = document.createElement('div');
    el.className = 'token-item';
    el.id = `token-el-${i}`;
    el.textContent = tok;
    el.title = `Token ${i + 1}: "${tok}"`;
    container.appendChild(el);
  });
}

// 2. QKV
function renderQkv() {
  const container = document.getElementById('col-qkv');
  if (!container) return;
  container.innerHTML = '';

  state.tokens.forEach((_, i) => {
    const grp = document.createElement('div');
    grp.className = 'qkv-cell';
    grp.id = `qkv-el-${i}`;
    grp.innerHTML = `
      <div class="q-badge">Q</div>
      <div class="k-badge">K</div>
      <div class="v-badge">V</div>
    `;
    container.appendChild(grp);
  });
}

// 3. Attention Card & Internal Views
function renderAttentionCard() {
  const matrix = state.attentions[state.currentBlock]?.[state.currentHead] || [];
  const n = state.tokens.length;

  // Single standard matrix
  const grid = document.getElementById('dot-matrix');
  if (grid) {
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${n}, 13px)`;
    grid.style.gridTemplateRows = `repeat(${n}, 13px)`;

    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const dot = document.createElement('div');
        dot.className = 'matrix-circle';
        const w = matrix[r]?.[c] ?? (c <= r ? 1 / (r + 1) : 0);

        if (c > r) {
          dot.style.background = '#f1f5f9';
          dot.style.opacity = '0.3';
        } else {
          dot.style.background = '#7c3aed';
          dot.style.opacity = Math.max(0.12, Math.min(1, w * 2.5));
        }

        dot.title = `Query: "${state.tokens[r]}" attends to Key: "${state.tokens[c]}" (${(w * 100).toFixed(1)}%)`;
        dot.addEventListener('mouseenter', () => highlightPair(r, c));
        dot.addEventListener('mouseleave', () => resetHighlights());
        grid.appendChild(dot);
      }
    }
  }

  // 3-Stage internal matrix views (Screenshot 3)
  renderSubMatrix('matrix-dotprod', n, (r, c) => (c <= r ? Math.random() : 0), '#8b5cf6');
  renderSubMatrix('matrix-mask', n, (r, c) => (c <= r ? Math.random() : 0), '#a855f7');
  renderSubMatrix('matrix-softmax', n, (r, c) => matrix[r]?.[c] ?? (c <= r ? 1 / (r + 1) : 0), '#7c3aed');

  // Stepper text
  const headText = document.getElementById('head-indicator-text');
  if (headText) {
    headText.textContent = `Head ${state.currentHead + 1} of 12`;
  }
}

function renderSubMatrix(id, n, calcVal, color) {
  const container = document.getElementById(id);
  if (!container) return;
  container.innerHTML = '';
  container.style.gridTemplateColumns = `repeat(${n}, 9px)`;
  container.style.gridTemplateRows = `repeat(${n}, 9px)`;

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const dot = document.createElement('div');
      dot.className = 'matrix-circle';
      dot.style.width = '9px';
      dot.style.height = '9px';
      const w = calcVal(r, c);

      if (c > r) {
        dot.style.background = '#f1f5f9';
        dot.style.opacity = '0.25';
      } else {
        dot.style.background = color;
        dot.style.opacity = Math.max(0.15, Math.min(1, w * 2.2));
      }
      container.appendChild(dot);
    }
  }
}

function highlightPair(q, k) {
  const qEl = document.getElementById(`token-el-${q}`);
  const kEl = document.getElementById(`token-el-${k}`);
  if (qEl) qEl.style.background = '#dbeafe';
  if (kEl) kEl.style.background = '#ffe4e6';
}

function resetHighlights() {
  document.querySelectorAll('.token-item').forEach(el => el.style.background = '');
}

// 4. MLP Column
function renderMlp() {
  const container = document.getElementById('col-mlp');
  if (!container) return;
  container.innerHTML = '';

  state.tokens.forEach(tok => {
    const el = document.createElement('div');
    el.className = 'mlp-token-item';
    el.textContent = tok;
    container.appendChild(el);
  });
}

// 5. Subsequent Blocks
function renderSubsequent() {
  const badge = document.getElementById('stream-token-badge');
  if (badge && state.tokens.length > 0) {
    badge.textContent = state.tokens[state.tokens.length - 1];
  }
}

// 6. Probabilities & Math Tooltip
function renderProbabilities() {
  const container = document.getElementById('probs-container');
  if (!container) return;
  container.innerHTML = '';

  const list = state.probabilities.slice(0, 14);
  const maxP = Math.max(...list.map(p => p.probability), 0.01);
  const sampledTok = cleanToken(state.sampled?.token);

  list.forEach(item => {
    const name = cleanToken(item.token);
    const isSampled = name === sampledTok || item.probability === state.sampled?.probability;
    const pct = (item.probability * 100).toFixed(2);
    const fill = maxP > 0 ? (item.probability / maxP) * 100 : 0;

    const row = document.createElement('div');
    row.className = `prob-item ${isSampled ? 'active-predicted' : ''}`;
    row.innerHTML = `
      <div class="prob-name">${name}</div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${Math.max(2, fill)}%"></div>
      </div>
      <div class="prob-num">${pct}%</div>
    `;

    // Interactive tooltip (Screenshot 4)
    row.addEventListener('mouseenter', () => showMathTooltip(item, name, pct));
    row.addEventListener('mouseleave', () => hideMathTooltip());
    container.appendChild(row);
  });
}

function showMathTooltip(item, name, pct) {
  const tip = document.getElementById('math-tooltip');
  if (!tip) return;

  const rawLogit = (item.logit ?? -135.91).toFixed(2);
  const temp = state.temperature.toFixed(1);
  const scaled = ((item.logit ?? -135.91) / state.temperature).toFixed(2);

  document.getElementById('math-card-title').textContent = `Probability of "${name}" token being sampled`;
  document.getElementById('math-raw-logit').textContent = rawLogit;
  document.getElementById('math-temp').textContent = temp;
  document.getElementById('math-scaled').textContent = scaled;
  document.getElementById('math-topk').textContent = scaled;
  document.getElementById('math-prob').textContent = `${pct}%`;

  tip.classList.remove('hidden');
}

function hideMathTooltip() {
  const tip = document.getElementById('math-tooltip');
  if (tip) tip.classList.add('hidden');
}

// 7. SVG Curved Bezier Ribbons
function renderRibbons() {
  const svg = document.getElementById('svg-ribbons');
  if (!svg) return;
  svg.innerHTML = '';

  const meshRect = document.querySelector('.flow-grid')?.getBoundingClientRect();
  if (!meshRect) return;

  // 1. Tokens to QKV
  state.tokens.forEach((_, i) => {
    const tEl = document.getElementById(`token-el-${i}`);
    const qEl = document.getElementById(`qkv-el-${i}`);
    if (tEl && qEl) {
      const tr = tEl.getBoundingClientRect();
      const qr = qEl.getBoundingClientRect();
      drawCurve(
        svg,
        tr.right - meshRect.left,
        tr.top + tr.height / 2 - meshRect.top,
        qr.left - meshRect.left,
        qr.top + qr.height / 2 - meshRect.top,
        'rgba(148, 163, 184, 0.35)',
        7
      );
    }
  });

  // 2. QKV to Attention Card & Over-the-top Residual
  const cardEl = document.querySelector('.card-foreground')?.getBoundingClientRect();
  if (cardEl) {
    const cardX = cardEl.left - meshRect.left;
    const cardY = cardEl.top - meshRect.top;

    state.tokens.forEach((_, i) => {
      const qEl = document.getElementById(`qkv-el-${i}`);
      if (!qEl) return;
      const qr = qEl.getBoundingClientRect();
      const x1 = qr.right - meshRect.left;
      const y1 = qr.top + qr.height / 2 - meshRect.top;

      // Key wires (pink)
      drawCurve(svg, x1, y1 - 4, cardX + 45, cardY + 20, 'rgba(244, 63, 94, 0.45)', 2.5);
      // Query wires (blue)
      drawCurve(svg, x1, y1, cardX + 15, cardY + 130 + i * 8, 'rgba(59, 130, 246, 0.45)', 2.5);
      // Value wires (green)
      drawCurve(svg, x1, y1 + 4, cardX + 65, cardY + 360, 'rgba(16, 185, 129, 0.45)', 2.5);
    });

    // Residual Over-the-top Loop
    const mlpEl = document.getElementById('col-mlp')?.getBoundingClientRect();
    if (mlpEl) {
      const mlpX = mlpEl.left - meshRect.left;
      drawArch(svg, cardX - 50, 10, mlpX + 20, 10, 'rgba(148, 163, 184, 0.4)');

      // Out stream from Attention Card to MLP
      const outX = cardEl.right - meshRect.left;
      const outY = cardEl.top + cardEl.height / 2 - meshRect.top;
      state.tokens.forEach((_, i) => {
        const mTok = document.querySelectorAll('.mlp-token-item')[i];
        if (mTok) {
          const mr = mTok.getBoundingClientRect();
          drawCurve(svg, outX, outY, mlpX, mr.top + mr.height / 2 - meshRect.top, 'rgba(124, 58, 237, 0.35)', 4);
        }
      });
    }
  }
}

function drawCurve(svg, x1, y1, x2, y2, color, strokeWidth = 3) {
  const dx = Math.abs(x2 - x1) * 0.55;
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', color);
  path.setAttribute('stroke-width', strokeWidth);
  path.setAttribute('stroke-linecap', 'round');
  svg.appendChild(path);
}

function drawArch(svg, x1, y1, x2, y2, color) {
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  const midX = (x1 + x2) / 2;
  path.setAttribute('d', `M ${x1} ${y1 + 18} Q ${midX} ${y1 - 18}, ${x2} ${y2 + 18}`);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', color);
  path.setAttribute('stroke-width', 2);
  path.setAttribute('stroke-dasharray', '4 4');
  svg.appendChild(path);
}

// ── Interactive Modals & Toggles ────────────────────────────
function toggleThreeStageView() {
  state.isThreeStageOpen = !state.isThreeStageOpen;
  document.getElementById('standard-matrix-wrap')?.classList.toggle('hidden', state.isThreeStageOpen);
  document.getElementById('three-stage-wrap')?.classList.toggle('hidden', !state.isThreeStageOpen);
}

function toggleProbDetails() {
  state.isProbDetailsOpen = !state.isProbDetailsOpen;
  document.getElementById('prob-subheaders')?.classList.toggle('hidden', !state.isProbDetailsOpen);
}

function toggleHeatmapModal() {
  const modal = document.getElementById('heatmap-modal');
  if (modal) {
    modal.classList.toggle('hidden');
    if (!modal.classList.contains('hidden')) {
      renderHeatmapTable();
    }
  }
}

function openHeatmapModal() {
  toggleHeatmapModal();
}

function initHeatmapSelectors() {
  const bSel = document.getElementById('select-modal-block');
  const hSel = document.getElementById('select-modal-head');
  if (bSel) {
    bSel.innerHTML = Array.from({ length: 12 }, (_, i) => `<option value="${i}">Block ${i + 1}</option>`).join('');
    bSel.value = state.currentBlock;
  }
  if (hSel) {
    hSel.innerHTML = Array.from({ length: 12 }, (_, i) => `<option value="${i}">Head ${i + 1}</option>`).join('');
    hSel.value = state.currentHead;
  }
}

function setModalBlock(val) {
  state.currentBlock = parseInt(val);
  renderHeatmapTable();
  renderAttentionCard();
}

function setModalHead(val) {
  state.currentHead = parseInt(val);
  renderHeatmapTable();
  renderAttentionCard();
}

function renderHeatmapTable() {
  const container = document.getElementById('heatmap-table-container');
  const title = document.getElementById('modal-title');
  if (!container) return;

  if (title) {
    title.textContent = `Self-Attention Matrix Heatmap — Block ${state.currentBlock + 1}, Head ${state.currentHead + 1}`;
  }

  const n = state.tokens.length;
  const matrix = state.attentions[state.currentBlock]?.[state.currentHead] || [];

  let html = `<table class="heatmap-table"><tr><th></th>`;
  state.tokens.forEach(t => html += `<th>${t}</th>`);
  html += `</tr>`;

  for (let r = 0; r < n; r++) {
    html += `<tr><th style="text-align:right;">${state.tokens[r]}</th>`;
    for (let c = 0; c < n; c++) {
      const w = matrix[r]?.[c] ?? (c <= r ? 1 / (r + 1) : 0);
      if (c <= r) {
        const pct = (w * 100).toFixed(0) + '%';
        const bg = `rgba(124, 58, 237, ${Math.max(0.12, w)})`;
        const col = w > 0.4 ? '#ffffff' : '#1e293b';
        html += `<td style="background:${bg};color:${col};border-radius:4px;">${pct}</td>`;
      } else {
        html += `<td style="background:#f8fafc;color:#cbd5e1;">—</td>`;
      }
    }
    html += `</tr>`;
  }
  html += `</table>`;
  container.innerHTML = html;
}

// ── Controls ────────────────────────────────────────────────
function prevHead() {
  state.currentHead = (state.currentHead - 1 + 12) % 12;
  renderAttentionCard();
}
function nextHead() {
  state.currentHead = (state.currentHead + 1) % 12;
  renderAttentionCard();
}
function prevBlock() {
  if (state.currentBlock > 0) { state.currentBlock--; renderAttentionCard(); }
}
function nextBlock() {
  if (state.currentBlock < 11) { state.currentBlock++; renderAttentionCard(); }
}

function setTemp(v) {
  state.temperature = parseFloat(v);
  document.getElementById('temp-val').textContent = state.temperature.toFixed(2);
}
function setSamp(t) {
  state.samplingType = t;
  const lbl = document.getElementById('samp-label');
  const inp = document.getElementById('samp-input');
  if (t === 'top-p') {
    lbl.textContent = 'p=';
    inp.min = '0.05'; inp.max = '1.0'; inp.step = '0.05'; inp.value = '0.9';
    state.samplingValue = 0.9;
  } else {
    lbl.textContent = 'k=';
    inp.min = '1'; inp.max = '50'; inp.step = '1'; inp.value = '5';
    state.samplingValue = 5;
  }
}
function setSampVal(v) { state.samplingValue = parseFloat(v); }

function toggleExamples(e) {
  e.stopPropagation();
  document.getElementById('examples-menu')?.classList.toggle('hidden');
}
function pickExample(idx) {
  const txt = EXAMPLES[idx] || EXAMPLES[0];
  document.getElementById('input-text-display').innerText = txt;
  state.text = txt;
  document.getElementById('examples-menu')?.classList.add('hidden');
  generatePrediction();
}

function handleInputKey(e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    generatePrediction();
  }
}

function setLoadingState(on) {
  const btn = document.getElementById('generate-label');
  if (btn) btn.textContent = on ? 'Running…' : 'Generate';
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}
