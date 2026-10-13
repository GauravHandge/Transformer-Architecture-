/* ─────────────────────────────────────────────────────────────
   Transformer Explainer — app.js
   Connects HTML frontend to Python FastAPI backend.
   Uses D3.js for all visualizations.
───────────────────────────────────────────────────────────── */

const API = 'http://127.0.0.1:8000/api';

// ── State ──────────────────────────────────────────────────
let state = {
  tokens: [],
  tokenIds: [],
  attentions: [],     // [layer][head][seq][seq]
  hiddenStates: [],
  probabilities: [],
  sampled: null,
  currentBlock: 0,
  textbookPage: 'overview',
  isLoaded: false,
  isRunning: false,
};

const EXAMPLES = [
  'Data visualization empowers users to',
  'Artificial Intelligence is transforming the',
  'As the spaceship was approaching the',
  'On the deserted planet they discovered a',
  'IEEE VIS conference highlights the',
];

const GPT2_LAYERS = 12;
const GPT2_HEADS  = 12;

// ── Init ────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  checkBackend();
  populateSelects();
  showTextbookPage('overview');
});

async function checkBackend() {
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      hideLoading();
      state.isLoaded = true;
      setStatusDot('ready');
      document.querySelector('#topbar .model-label').textContent = 'Model Ready';
      // auto-run first example
      await runModel();
    } else { throw new Error('Backend error'); }
  } catch (e) {
    document.querySelector('#loading-overlay p').textContent = '❌ Backend not running';
    document.querySelector('.loading-sub').textContent =
      'Please start: uvicorn backend.main:app --reload';
    setStatusDot('error');
  }
}

function hideLoading() {
  document.getElementById('loading-overlay').style.display = 'none';
}
function setStatusDot(state) {
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot ' + state;
}

function populateSelects() {
  const layerSel = document.getElementById('attn-layer');
  const headSel  = document.getElementById('attn-head');
  for (let i = 0; i < GPT2_LAYERS; i++) {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = `Layer ${i + 1}`;
    layerSel.appendChild(opt);
  }
  for (let i = 0; i < GPT2_HEADS; i++) {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = `Head ${i + 1}`;
    headSel.appendChild(opt);
  }
}

// ── Run Model ─────────────────────────────────────────────
async function runModel() {
  if (state.isRunning) return;
  state.isRunning = true;

  const text   = document.getElementById('input-text').value;
  const temp   = parseFloat(document.getElementById('temperature').value);
  const sType  = document.getElementById('sampling-type').value;
  const sValue = parseFloat(document.getElementById('sampling-value').value);

  setRunBtnLoading(true);
  setStatusDot('loading');

  try {
    const res = await fetch(`${API}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        temperature: temp,
        sampling_type: sType,
        sampling_value: sValue,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert('Error: ' + (err.detail || 'Unknown error'));
      return;
    }

    const data = await res.json();
    state.tokens       = data.tokens;
    state.tokenIds     = data.tokenIds;
    state.attentions   = data.attentions;
    state.hiddenStates = data.hiddenStates;
    state.probabilities = data.probabilities;
    state.sampled      = data.sampled;
    state.currentBlock = 0;

    renderAll();
    setStatusDot('ready');
  } catch (e) {
    console.error(e);
    setStatusDot('error');
    alert('Cannot connect to backend. Make sure uvicorn is running.');
  } finally {
    setRunBtnLoading(false);
    state.isRunning = false;
  }
}

function setRunBtnLoading(on) {
  const btn    = document.getElementById('run-btn');
  const label  = document.getElementById('run-btn-label');
  const spin   = document.getElementById('run-spinner');
  btn.disabled = on;
  label.classList.toggle('hidden', on);
  spin.classList.toggle('hidden', !on);
}

// ── Render All ────────────────────────────────────────────
function renderAll() {
  renderTokens();
  renderEmbeddings();
  renderAttentionMini();
  renderProbBars();
  renderPredictedToken();
  updateBlockIndicator();
}

// ── Tokens ────────────────────────────────────────────────
function renderTokens() {
  const container = document.getElementById('tokens-display');
  container.innerHTML = '';
  state.tokens.forEach(tok => {
    const chip = document.createElement('div');
    chip.className = 'token-chip fade-in';
    chip.textContent = tok || '·';
    chip.title = `"${tok}"`;
    container.appendChild(chip);
  });
}

// ── Embeddings (visual bars using hidden state values) ────
function renderEmbeddings() {
  const container = document.getElementById('embedding-display');
  container.innerHTML = '';
  if (!state.hiddenStates.length) return;

  const emb = state.hiddenStates[0]; // embedding layer output
  const colors = ['#7c3aed', '#2563eb', '#4f46e5', '#0891b2', '#059669'];

  emb.forEach((vec, i) => {
    // Use first 8 dims normalized as visual bar
    const vals = vec.slice(0, 8);
    const max  = Math.max(...vals.map(Math.abs)) || 1;
    const norm = vals.map(v => Math.abs(v) / max);
    const avg  = norm.reduce((a, b) => a + b, 0) / norm.length;

    const wrap = document.createElement('div');
    wrap.className = 'emb-bar-wrap fade-in';

    const bar = document.createElement('div');
    bar.className = 'emb-bar';
    bar.style.width = Math.round(avg * 80) + '%';
    bar.style.background = `linear-gradient(90deg, ${colors[i % colors.length]}, #a78bfa)`;

    const label = document.createElement('span');
    label.className = 'emb-label';
    label.textContent = state.tokens[i] || `t${i}`;

    wrap.appendChild(bar);
    wrap.appendChild(label);
    container.appendChild(wrap);
  });
}

// ── Attention Mini Preview ─────────────────────────────────
function renderAttentionMini() {
  if (!state.attentions.length) return;

  const container = document.getElementById('attn-mini-preview');
  container.innerHTML = '';

  const layer = state.currentBlock;
  const head  = 0;
  const matrix = state.attentions[layer]?.[head];
  if (!matrix) return;

  const size = matrix.length;
  const cellSize = Math.min(8, Math.floor(60 / size));
  const svgW = size * cellSize;
  const svgH = size * cellSize;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', svgW)
    .attr('height', svgH);

  const color = d3.scaleSequential(d3.interpolatePurples).domain([0, 1]);

  matrix.forEach((row, r) => {
    row.forEach((val, c) => {
      svg.append('rect')
        .attr('x', c * cellSize)
        .attr('y', r * cellSize)
        .attr('width', cellSize - 1)
        .attr('height', cellSize - 1)
        .attr('rx', 1)
        .attr('fill', isFinite(val) && val > 0 ? color(val) : '#f3f4f6');
    });
  });
}

// ── Full Attention Matrix ─────────────────────────────────
function renderAttention() {
  if (!state.attentions.length) return;

  const layer  = parseInt(document.getElementById('attn-layer').value);
  const head   = parseInt(document.getElementById('attn-head').value);
  document.getElementById('attn-layer-label').textContent = layer + 1;
  document.getElementById('attn-head-label').textContent  = head + 1;

  const matrix = state.attentions[layer]?.[head];
  if (!matrix) return;

  const tokens = state.tokens;
  const n      = matrix.length;
  const cellSize = 36;
  const margin   = { top: 20, left: 80, right: 20, bottom: 20 };
  const svgW     = margin.left + n * cellSize + margin.right;
  const svgH     = margin.top  + n * cellSize + margin.bottom;

  const svgEl = document.getElementById('attn-matrix-svg');
  svgEl.setAttribute('width',  svgW);
  svgEl.setAttribute('height', svgH);
  svgEl.innerHTML = '';

  const svg = d3.select(svgEl);
  const g   = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const color = d3.scaleSequential(d3.interpolatePurples).domain([0, 1]);

  // Cells
  matrix.forEach((row, r) => {
    row.forEach((val, c) => {
      const v = isFinite(val) && val > 0 ? val : 0;
      g.append('rect')
        .attr('x', c * cellSize)
        .attr('y', r * cellSize)
        .attr('width', cellSize - 2)
        .attr('height', cellSize - 2)
        .attr('rx', 3)
        .attr('fill', v > 0 ? color(v) : '#f9fafb')
        .attr('stroke', '#e5e7eb')
        .attr('stroke-width', 0.5)
        .append('title')
        .text(`${tokens[r] || r} → ${tokens[c] || c}: ${v.toFixed(3)}`);

      if (v > 0.15) {
        g.append('text')
          .attr('x', c * cellSize + cellSize / 2 - 1)
          .attr('y', r * cellSize + cellSize / 2 + 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', '8px')
          .attr('fill', v > 0.5 ? 'white' : '#374151')
          .text(v.toFixed(2));
      }
    });
  });

  // Row labels (Y-axis)
  tokens.forEach((tok, i) => {
    g.append('text')
      .attr('x', -6)
      .attr('y', i * cellSize + cellSize / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', '10px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('fill', '#6b7280')
      .text((tok || `T${i}`).slice(0, 10));
  });

  // Col labels (X-axis)
  tokens.forEach((tok, i) => {
    g.append('text')
      .attr('x', i * cellSize + cellSize / 2)
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('fill', '#6b7280')
      .text((tok || `T${i}`).slice(0, 6));
  });
}

// ── Probability Bars ──────────────────────────────────────
function renderProbBars() {
  const container = document.getElementById('prob-bars');
  container.innerHTML = '';
  if (!state.probabilities.length) return;

  const sampledId = state.sampled?.tokenId;
  const topN = state.probabilities.slice(0, 15);
  const maxProb = Math.max(...topN.map(p => p.probability));

  topN.forEach(item => {
    const isSampled = item.tokenId === sampledId;
    const pct = maxProb > 0 ? (item.probability / maxProb) * 100 : 0;

    const row = document.createElement('div');
    row.className = 'prob-row';

    const tokLabel = document.createElement('div');
    tokLabel.className = 'prob-token';
    tokLabel.textContent = item.token || `id:${item.tokenId}`;
    tokLabel.title = `Token: "${item.token}" | ID: ${item.tokenId}`;

    const barBg = document.createElement('div');
    barBg.className = 'prob-bar-bg';

    const barFill = document.createElement('div');
    barFill.className = 'prob-bar-fill' + (isSampled ? ' sampled' : '');
    barFill.style.width = pct + '%';

    const pctLabel = document.createElement('div');
    pctLabel.className = 'prob-pct';
    pctLabel.textContent = (item.probability * 100).toFixed(1) + '%';

    barBg.appendChild(barFill);
    row.appendChild(tokLabel);
    row.appendChild(barBg);
    row.appendChild(pctLabel);
    container.appendChild(row);
  });
}

// ── Predicted Token ───────────────────────────────────────
function renderPredictedToken() {
  const el = document.getElementById('predicted-token');
  if (state.sampled) {
    el.textContent = state.sampled.token || `[id:${state.sampled.tokenId}]`;
    el.classList.add('fade-in');
  } else {
    el.textContent = '—';
  }
}

// ── Block Navigation ──────────────────────────────────────
function changeBlock(dir) {
  const next = state.currentBlock + dir;
  if (next < 0 || next >= GPT2_LAYERS) return;
  state.currentBlock = next;
  updateBlockIndicator();
  renderAttentionMini();
  // Update full attention matrix if panel open
  if (!document.getElementById('attention-panel').classList.contains('hidden')) {
    document.getElementById('attn-layer').value = state.currentBlock;
    renderAttention();
  }
}

function updateBlockIndicator() {
  document.getElementById('block-indicator').textContent =
    `Block ${state.currentBlock + 1} / ${GPT2_LAYERS}`;
  document.getElementById('prev-block').disabled = state.currentBlock === 0;
  document.getElementById('next-block').disabled = state.currentBlock === GPT2_LAYERS - 1;
}

// ── Panels ─────────────────────────────────────────────────
function togglePanel(id) {
  const panel = document.getElementById(id);
  const isHidden = panel.classList.contains('hidden');
  panel.classList.toggle('hidden');
  if (isHidden) {
    document.getElementById('attn-layer').value = state.currentBlock;
    renderAttention();
  }
}

// ── Textbook ──────────────────────────────────────────────
const TEXTBOOK_PAGES = {
  overview: `
    <h3>What is a Transformer?</h3>
    <p>A Transformer is a deep learning model that processes sequences using <strong>Self-Attention</strong>. GPT-2 has 12 Transformer blocks stacked on top of each other.</p>
    <p>Each block contains:<br>• Self-Attention layer<br>• MLP (Feed-Forward) layer<br>• Layer Normalization<br>• Residual connections</p>
    <div class="math-formula">Input → Embed → [Block×12] → Linear → Softmax → Token</div>
  `,
  tokenization: `
    <h3>Tokenization</h3>
    <p>Text is first split into <strong>tokens</strong> — subword units. GPT-2 uses Byte-Pair Encoding (BPE) with a vocabulary of 50,257 tokens.</p>
    <p>Example: "Hello world" → ["Hello", " world"]</p>
    <div class="math-formula">text → tokenizer.encode(text) → [id₁, id₂, …, idₙ]</div>
    <p>Each token ID maps to a row in the embedding matrix.</p>
  `,
  embedding: `
    <h3>Embedding</h3>
    <p>Each token ID is converted into a 768-dimensional vector. Two embeddings are summed:</p>
    <p>1. <strong>Token Embedding</strong> — learned representation of each token<br>2. <strong>Position Embedding</strong> — encodes position in sequence</p>
    <div class="math-formula">x = TokenEmb[id] + PosEmb[position]</div>
  `,
  attention: `
    <h3>Self-Attention</h3>
    <p>Self-Attention lets each token "look at" other tokens to build context. GPT-2 uses 12 attention heads in parallel.</p>
    <div class="math-formula">Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V</div>
    <p>• <strong>Q (Query)</strong> — what to look for<br>• <strong>K (Key)</strong> — what each token offers<br>• <strong>V (Value)</strong> — information to aggregate</p>
    <p>GPT-2 uses <strong>causal masking</strong> so tokens can only attend to previous tokens.</p>
  `,
  mlp: `
    <h3>MLP / Feed-Forward Network</h3>
    <p>After attention, each token is processed independently by a 2-layer MLP:</p>
    <div class="math-formula">FFN(x) = GELU(xW₁ + b₁)W₂ + b₂</div>
    <p>• Hidden dimension: 768 → 3072 → 768<br>• Expansion factor: ×4<br>• Activation: GELU (Gaussian Error Linear Unit)</p>
  `,
  softmax: `
    <h3>Linear + Softmax Output</h3>
    <p>The final hidden state of the last token is projected to vocabulary size (50,257) and converted to probabilities:</p>
    <div class="math-formula">P(next token) = softmax(hW + b)</div>
    <p>• <strong>Temperature</strong> controls sharpness (↑ temp = more random)<br>• <strong>Top-K</strong> keeps only K highest probability tokens<br>• <strong>Top-P</strong> keeps tokens summing to probability p</p>
  `,
};

function showTextbookPage(page) {
  state.textbookPage = page;
  document.getElementById('textbook-content').innerHTML = TEXTBOOK_PAGES[page] || '';
  document.querySelectorAll('.tb-item').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('onclick') === `showTextbookPage('${page}')`);
  });
}

function toggleTextbook() {
  document.getElementById('textbook-panel').classList.toggle('open');
}

// ── Examples ─────────────────────────────────────────────
function loadExample(idx) {
  document.getElementById('input-text').value = EXAMPLES[idx];
  runModel();
}

// ── Temperature Label ────────────────────────────────────
function updateTempLabel(val) {
  document.getElementById('temp-label').textContent = parseFloat(val).toFixed(2);
}

// ── Sampling UI ──────────────────────────────────────────
function updateSamplingUI() {
  const type = document.getElementById('sampling-type').value;
  const label = document.getElementById('sampling-value-label');
  if (type === 'top-k') {
    label.innerHTML = 'K = <input id="sampling-value" type="number" min="1" max="50" value="5" style="width:4rem" />';
  } else {
    label.innerHTML = 'P = <input id="sampling-value" type="number" min="0.01" max="1" step="0.05" value="0.9" style="width:4rem" />';
  }
}

// ── Enter key runs model ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('input-text').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) runModel();
  });
});
