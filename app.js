/**
 * app.js — UI controller for Warehouse Robot MDP prototype
 */

'use strict';

// Destructure engine
const {
  ROWS, COLS, GAMMA, CELL, GRID, REWARD, ACTIONS, ACTION_LABEL, ACTION_EMOJI,
  isTerminal, getTransitions,
  MDP, Episode,
} = window.MDP_ENGINE;

/* ============================================================
   STATE
   ============================================================ */

let mdp = new MDP();
let currentEpisode = null;
let simInterval = null;
let viInterval = null;
let currentPolicyType = 'optimal';

// Run value iteration to convergence on boot (for policy grid / simulator)
const bootIters = mdp.runToConvergence();

/* ============================================================
   UTILITY
   ============================================================ */

function cellClass(r, c) {
  const t = GRID[r][c];
  if (t === CELL.START) return 'cell-start';
  if (t === CELL.GOAL)  return 'cell-goal';
  if (t === CELL.TRAP)  return 'cell-trap';
  if (t === CELL.WALL)  return 'cell-wall';
  if (t === CELL.DOCK)  return 'cell-dock';
  return 'cell-open';
}

function cellIcon(r, c) {
  const t = GRID[r][c];
  if (t === CELL.START) return '🤖';
  if (t === CELL.GOAL)  return '📦';
  if (t === CELL.TRAP)  return '⚠️';
  if (t === CELL.WALL)  return '🧱';
  if (t === CELL.DOCK)  return '⚡';
  return '';
}

function cellLabel(r, c) {
  const t = GRID[r][c];
  if (t === CELL.START) return 'Start';
  if (t === CELL.GOAL)  return 'Goal';
  if (t === CELL.TRAP)  return 'Trap';
  if (t === CELL.WALL)  return 'Wall';
  if (t === CELL.DOCK)  return 'Dock';
  return '';
}

function fmt(v) {
  if (v === null || v === undefined) return '—';
  return (typeof v === 'number') ? v.toFixed(2) : v;
}

function valueColor(v, minV, maxV) {
  if (v === null) return 'rgba(255,255,255,0.04)';
  const norm = maxV === minV ? 0.5 : (v - minV) / (maxV - minV);
  // low = red (#ef4444), mid = amber (#f59e0b), high = green (#10b981) / cyan
  if (norm < 0.5) {
    // red → amber
    const t = norm * 2;
    return `hsl(${Math.round(0 + t * 38)}, 85%, ${20 + t * 8}%)`;
  } else {
    // amber → cyan-green
    const t = (norm - 0.5) * 2;
    return `hsl(${Math.round(38 + t * 127)}, 80%, ${28 + t * 6}%)`;
  }
}

/* ============================================================
   STATIC GRID (MDP section)
   ============================================================ */

function buildStaticGrid() {
  const container = document.getElementById('static-grid');
  container.innerHTML = '';
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement('div');
      cell.className = `grid-cell ${cellClass(r, c)}`;
      cell.innerHTML = `
        <span class="cell-icon">${cellIcon(r, c)}</span>
        ${cellLabel(r, c) ? `<span class="cell-label">${cellLabel(r, c)}</span>` : ''}
        <span class="cell-coord">(${r},${c})</span>
      `;
      container.appendChild(cell);
    }
  }
}

/* ============================================================
   VALUE GRID (Bellman section)
   ============================================================ */

function buildValueGrid(V) {
  const container = document.getElementById('value-grid');
  container.innerHTML = '';

  // Compute min/max for coloring (exclude walls)
  let minV = Infinity, maxV = -Infinity;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (GRID[r][c] !== CELL.WALL) {
        minV = Math.min(minV, V[r][c]);
        maxV = Math.max(maxV, V[r][c]);
      }
    }
  }

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement('div');
      cell.className = 'value-cell';
      cell.id = `vc-${r}-${c}`;

      const isWall = GRID[r][c] === CELL.WALL;
      const v = V[r][c];
      cell.style.background = isWall ? 'rgba(239,68,68,0.1)' : valueColor(v, minV, maxV);
      cell.style.border = `1px solid ${isWall ? 'rgba(239,68,68,0.2)' : 'rgba(255,255,255,0.06)'}`;

      const icon = cellIcon(r, c);
      cell.innerHTML = `
        <span class="vc-coord">(${r},${c})</span>
        ${icon ? `<span style="font-size:1rem">${icon}</span>` : ''}
        <span class="vc-val">${isWall ? '—' : fmt(v)}</span>
      `;

      cell.title = `State (${r},${c}): V* = ${fmt(v)}`;
      container.appendChild(cell);
    }
  }
}

/* ============================================================
   VALUE ITERATION CONTROLS
   ============================================================ */

let viResetDone = false;

function updateVIDisplay(delta) {
  document.getElementById('vi-iter').textContent = mdp.iteration;
  document.getElementById('vi-delta').textContent = delta !== null ? delta.toFixed(4) : '—';
  document.getElementById('vi-status').textContent = mdp.converged ? '✅ Converged' : 'Running…';
  document.getElementById('vg-iter-label').textContent = mdp.iteration;
  buildValueGrid(mdp.V);
  drawConvergenceChart();

  if (mdp.converged) {
    updateBellmanWorkedExample();
    populateValueTable();
    document.getElementById('conv-iters').textContent = mdp.iteration;
  }
}

document.getElementById('btn-run-vi').addEventListener('click', () => {
  if (mdp.converged) return;
  if (viInterval) clearInterval(viInterval);

  viInterval = setInterval(() => {
    const delta = mdp.step();
    updateVIDisplay(delta);
    if (mdp.converged) {
      clearInterval(viInterval);
      viInterval = null;
      // Also refresh sim policy and policy map
      buildPolicyGrid(currentPolicyType);
      resetSimulator();
    }
  }, 60);
});

document.getElementById('btn-step-vi').addEventListener('click', () => {
  if (mdp.converged) return;
  const delta = mdp.step();
  updateVIDisplay(delta);
  if (mdp.converged) {
    buildPolicyGrid(currentPolicyType);
    resetSimulator();
  }
});

document.getElementById('btn-reset-vi').addEventListener('click', () => {
  if (viInterval) { clearInterval(viInterval); viInterval = null; }
  mdp.reset();
  updateVIDisplay(null);
  document.getElementById('vi-status').textContent = 'Reset';
  document.getElementById('value-table-body').innerHTML =
    '<tr><td colspan="7" style="text-align:center;color:var(--clr-muted)">Run Value Iteration to populate…</td></tr>';
  document.getElementById('bellman-result').innerHTML =
    '<strong>Result:</strong> Run Value Iteration below to see converged V* values.';
});

/* ============================================================
   CONVERGENCE CHART
   ============================================================ */

function drawConvergenceChart() {
  const canvas = document.getElementById('convergence-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 800;
  const H = canvas.offsetHeight || 220;
  canvas.width = W;
  canvas.height = H;

  const deltas = mdp.deltaHistory;
  if (deltas.length === 0) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(148,163,184,0.3)';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Run Value Iteration to see convergence plot', W / 2, H / 2);
    return;
  }

  ctx.clearRect(0, 0, W, H);

  const padL = 60, padR = 20, padT = 20, padB = 40;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxD = Math.max(...deltas) || 1;
  const minD = 0;

  // Background grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
  }

  // Gradient fill
  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, 'rgba(99,102,241,0.4)');
  grad.addColorStop(1, 'rgba(99,102,241,0)');

  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  for (let i = 0; i < deltas.length; i++) {
    const x = padL + (i / Math.max(deltas.length - 1, 1)) * plotW;
    const y = padT + plotH - ((deltas[i] - minD) / (maxD - minD)) * plotH;
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#6366f1';
  ctx.lineWidth = 2.5;
  for (let i = 0; i < deltas.length; i++) {
    const x = padL + (i / Math.max(deltas.length - 1, 1)) * plotW;
    const y = padT + plotH - ((deltas[i] - minD) / (maxD - minD)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Threshold line
  if (maxD > 0.001) {
    const threshY = padT + plotH - ((0.001 - minD) / (maxD - minD)) * plotH;
    if (threshY > padT && threshY < padT + plotH) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(16,185,129,0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.moveTo(padL, threshY);
      ctx.lineTo(padL + plotW, threshY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(16,185,129,0.8)';
      ctx.font = '10px Inter';
      ctx.fillText('ε = 0.001', padL + plotW - 60, threshY - 4);
    }
  }

  // Axes
  ctx.strokeStyle = 'rgba(148,163,184,0.3)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // Y labels
  ctx.fillStyle = 'rgba(148,163,184,0.7)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = maxD - (maxD * i) / 4;
    const y = padT + (plotH * i) / 4;
    ctx.fillText(val.toFixed(2), padL - 6, y + 4);
  }

  // X label
  ctx.textAlign = 'center';
  ctx.fillText('Iterations', padL + plotW / 2, H - 6);

  // Y label
  ctx.save();
  ctx.translate(14, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Max |ΔV|', 0, 0);
  ctx.restore();
}

/* ============================================================
   BELLMAN WORKED EXAMPLE
   ============================================================ */

function updateBellmanWorkedExample() {
  const r = 2, c = 2;
  const V = mdp.V;
  const Qvals = mdp.Qvals;

  if (!Qvals) return;

  const qRight = mdp.computeQ(r, c, 'right');
  const qUp = mdp.computeQ(r, c, 'up');
  const qDown = mdp.computeQ(r, c, 'down');
  const qLeft = mdp.computeQ(r, c, 'left');
  const vStar = Math.max(qRight, qUp, qDown, qLeft);

  // Detailed expansion for Right
  const v23 = V[2][3].toFixed(2);
  const v12 = V[1][2].toFixed(2); // trap = -50
  const v32 = V[3][2].toFixed(2);

  const term1 = (0.80 * (-1 + GAMMA * V[2][3])).toFixed(2);
  const term2 = (0.10 * (-50 + GAMMA * V[1][2])).toFixed(2);
  const term3 = (0.10 * (-1 + GAMMA * V[3][2])).toFixed(2);

  const resultEl = document.getElementById('bellman-result');
  resultEl.innerHTML = `
    <strong>Expansion of Q((2,2), Right):</strong><br>
    = 0.80 × [-1 + 0.9 × ${v23}] + 0.10 × [-50 + 0.9 × ${v12}] + 0.10 × [-1 + 0.9 × ${v32}]<br>
    = 0.80 × [${(-1 + GAMMA * V[2][3]).toFixed(2)}] + 0.10 × [${(-50 + GAMMA * V[1][2]).toFixed(2)}] + 0.10 × [${(-1 + GAMMA * V[3][2]).toFixed(2)}]<br>
    = <strong>${term1}</strong> + <strong>${term2}</strong> + <strong>${term3}</strong><br>
    = <strong style="color:var(--clr-cyan)">${qRight.toFixed(2)}</strong><br><br>
    All Q-values: ↑ ${qUp.toFixed(2)} &nbsp;↓ ${qDown.toFixed(2)} &nbsp;← ${qLeft.toFixed(2)} &nbsp;→ ${qRight.toFixed(2)}<br>
    <span style="color:var(--clr-green)">V*(2,2) = max(Q) = <strong>${vStar.toFixed(2)}</strong> via action: <strong>${ACTION_LABEL[mdp.policy[2][2]]}</strong></span>
  `;
}

/* ============================================================
   VALUE TABLE (Policy section)
   ============================================================ */

function populateValueTable() {
  const tbody = document.getElementById('value-table-body');
  tbody.innerHTML = '';

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const tr = document.createElement('tr');
      const type = GRID[r][c];
      const isWall = type === CELL.WALL;
      const isTerm = isTerminal(r, c);

      const policy = mdp.policy ? mdp.policy[r][c] : null;
      const qv = mdp.Qvals ? mdp.Qvals[r][c] : null;
      const v = mdp.V[r][c];

      const actionStr = policy ? `${ACTION_LABEL[policy]} ${policy}` : (isTerm ? 'TERMINAL' : (isWall ? 'WALL' : '—'));

      tr.innerHTML = `
        <td>(${r},${c}) ${type !== CELL.OPEN && type !== CELL.START ? `<small style="color:var(--clr-amber)">[${type}]</small>` : ''}</td>
        <td style="color:${v > 0 ? 'var(--clr-green)' : v < -5 ? 'var(--clr-red)' : 'var(--text-primary)'}"><strong>${fmt(v)}</strong></td>
        <td>${actionStr}</td>
        <td>${qv ? fmt(qv['up']) : '—'}</td>
        <td>${qv ? fmt(qv['down']) : '—'}</td>
        <td>${qv ? fmt(qv['left']) : '—'}</td>
        <td>${qv ? fmt(qv['right']) : '—'}</td>
      `;

      if (type === CELL.GOAL) tr.classList.add('row-good');
      else if (type === CELL.TRAP) tr.classList.add('row-bad');
      else if (type === CELL.WALL) tr.style.opacity = '0.5';

      tbody.appendChild(tr);
    }
  }
}

/* ============================================================
   SIMULATOR
   ============================================================ */

let trail = new Set();

function buildSimGrid(episode) {
  const container = document.getElementById('sim-grid');
  container.innerHTML = '';

  const rRobot = episode ? episode.r : 0;
  const cRobot = episode ? episode.c : 0;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement('div');
      const isRobot = r === rRobot && c === cRobot;
      const isTrail = trail.has(`${r},${c}`) && !isRobot;

      cell.className = `grid-cell ${isRobot ? 'cell-robot' : cellClass(r, c)}${isTrail ? ' cell-trail' : ''}`;
      cell.id = `sc-${r}-${c}`;

      const icon = isRobot ? '🤖' : cellIcon(r, c);
      const vStr = mdp.V ? mdp.V[r][c].toFixed(1) : '';

      cell.innerHTML = `
        <span class="cell-icon">${icon}</span>
        ${cellLabel(r, c) ? `<span class="cell-label">${cellLabel(r, c)}</span>` : ''}
        <span class="cell-coord" style="font-size:0.5rem">${vStr}</span>
      `;

      container.appendChild(cell);
    }
  }
}

function resetSimulator() {
  if (simInterval) { clearInterval(simInterval); simInterval = null; }
  trail.clear();
  currentEpisode = new Episode(mdp, currentPolicyType);
  buildSimGrid(currentEpisode);
  updateSimStats(null);
  document.getElementById('ep-log').innerHTML = '<div class="log-entry">Press "Auto-Run Episode" or "Step" to begin.</div>';
  document.getElementById('ep-status').textContent = 'Ready';
  document.getElementById('ep-status').className = 'status-running';
  document.getElementById('ep-step').textContent = '0';
  document.getElementById('ep-return').textContent = '0';
  document.getElementById('ep-state').textContent = '(0,0)';
  document.getElementById('ep-action').textContent = '—';
  document.getElementById('ep-reward').textContent = '—';
  document.getElementById('ep-value').textContent = mdp.V ? fmt(mdp.V[0][0]) : '—';
  document.getElementById('bellman-snap').innerHTML = '<div class="snap-placeholder">Run a step to see live Bellman calculation…</div>';
}

function updateSimStats(stepInfo) {
  if (!currentEpisode) return;
  const ep = currentEpisode;

  document.getElementById('ep-step').textContent = ep.step;
  document.getElementById('ep-state').textContent = `(${ep.r},${ep.c})`;
  document.getElementById('ep-value').textContent = mdp.V ? fmt(mdp.V[ep.r][ep.c]) : '—';
  document.getElementById('ep-return').textContent = fmt(ep.totalReturn);

  if (stepInfo) {
    document.getElementById('ep-action').textContent = `${ACTION_LABEL[stepInfo.action]} ${stepInfo.action}${!stepInfo.intended ? ` (slipped: ${ACTION_LABEL[stepInfo.actualAction]})` : ''}`;
    document.getElementById('ep-reward').textContent = `${stepInfo.reward >= 0 ? '+' : ''}${stepInfo.reward}`;
  }

  if (ep.done) {
    const statusEl = document.getElementById('ep-status');
    if (ep.outcome === 'success') {
      statusEl.textContent = '✅ Goal Reached!';
      statusEl.className = 'status-done';
    } else if (ep.outcome === 'failure') {
      statusEl.textContent = '❌ Fell into Trap!';
      statusEl.className = 'status-fail';
    } else {
      statusEl.textContent = '⏱ Timeout';
      statusEl.className = '';
    }
  }
}

function showBellmanSnap(r, c, stepInfo) {
  const el = document.getElementById('bellman-snap');
  if (!mdp.Qvals || !mdp.Qvals[r] || !mdp.Qvals[r][c]) {
    el.innerHTML = '<div class="snap-placeholder">Value iteration not yet run.</div>';
    return;
  }

  const qvals = mdp.Qvals[r][c];
  if (!qvals) {
    el.innerHTML = `<div class="snap-placeholder">Terminal or wall state — no transitions.</div>`;
    return;
  }

  const bestA = mdp.policy[r][c];
  let html = `<strong>State (${r},${c}) — Policy: ${ACTION_LABEL[bestA] || '—'}</strong><br>`;
  for (const a of ACTIONS) {
    const q = qvals[a];
    const isBest = a === bestA;
    html += `<span style="color:${isBest ? 'var(--clr-green)' : 'var(--text-muted)'}">
      ${isBest ? '★' : ' '} Q(s,${ACTION_LABEL[a]}) = ${fmt(q)}
    </span><br>`;
  }
  if (stepInfo) {
    html += `<br>V*(${r},${c}) = <strong style="color:var(--clr-cyan)">${fmt(mdp.V[r][c])}</strong>`;
  }
  el.innerHTML = html;
}

function addLog(text, cls = '') {
  const log = document.getElementById('ep-log');
  const div = document.createElement('div');
  div.className = `log-entry ${cls}`;
  div.textContent = text;
  log.insertBefore(div, log.firstChild);
  // Keep only last 20
  while (log.children.length > 20) log.removeChild(log.lastChild);
}

function simStep(manualAction = null) {
  if (!currentEpisode || currentEpisode.done) return;

  const before = { r: currentEpisode.r, c: currentEpisode.c };
  const stepInfo = currentEpisode.takeStep(manualAction);
  if (!stepInfo) return;

  trail.add(`${before.r},${before.c}`);
  buildSimGrid(currentEpisode);
  updateSimStats(stepInfo);
  showBellmanSnap(before.r, before.c, stepInfo);

  const slipNote = !stepInfo.intended ? ` [slipped to ${ACTION_LABEL[stepInfo.actualAction]}]` : '';
  const logClass = stepInfo.reward >= 50 ? 'log-good' : stepInfo.reward <= -20 ? 'log-bad' : 'log-step';
  addLog(
    `Step ${stepInfo.step}: (${before.r},${before.c}) ${ACTION_LABEL[stepInfo.action]}${slipNote} → (${stepInfo.nr},${stepInfo.nc}) R=${stepInfo.reward >= 0 ? '+' : ''}${stepInfo.reward}`,
    logClass
  );

  if (currentEpisode.done) {
    if (simInterval) { clearInterval(simInterval); simInterval = null; }
    addLog(
      currentEpisode.outcome === 'success' ? `✅ GOAL! Total return: ${fmt(currentEpisode.totalReturn)}` :
      currentEpisode.outcome === 'failure' ? `❌ TRAP! Total return: ${fmt(currentEpisode.totalReturn)}` :
      `⏱ Timeout. Total return: ${fmt(currentEpisode.totalReturn)}`,
      currentEpisode.outcome === 'success' ? 'log-good' : 'log-bad'
    );
    document.getElementById('btn-sim-start').disabled = true;
    document.getElementById('btn-sim-step').disabled = true;
  }
}

// Auto run
document.getElementById('btn-sim-start').addEventListener('click', () => {
  if (currentEpisode && currentEpisode.done) {
    resetSimulator();
    return;
  }
  if (simInterval) { clearInterval(simInterval); simInterval = null; }
  simInterval = setInterval(() => simStep(), 400);
  addLog('Episode started (auto-run)…', 'log-step');
});

document.getElementById('btn-sim-step').addEventListener('click', () => simStep());

document.getElementById('btn-sim-reset').addEventListener('click', () => {
  document.getElementById('btn-sim-start').disabled = false;
  document.getElementById('btn-sim-step').disabled = false;
  resetSimulator();
});

// Manual toggle
document.getElementById('manual-toggle').addEventListener('change', function () {
  const manual = this.checked;
  document.getElementById('manual-controls').style.display = manual ? 'flex' : 'none';
  document.getElementById('sim-mode-badge').textContent = manual ? 'Policy: Manual' : `Policy: ${currentPolicyType.charAt(0).toUpperCase() + currentPolicyType.slice(1)}`;
  if (simInterval) { clearInterval(simInterval); simInterval = null; }
});

// Direction buttons
document.getElementById('btn-up').addEventListener('click', () => simStep('up'));
document.getElementById('btn-down').addEventListener('click', () => simStep('down'));
document.getElementById('btn-left').addEventListener('click', () => simStep('left'));
document.getElementById('btn-right').addEventListener('click', () => simStep('right'));

// Keyboard controls
document.addEventListener('keydown', (e) => {
  if (!document.getElementById('manual-toggle').checked) return;
  const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
  if (map[e.key]) {
    e.preventDefault();
    simStep(map[e.key]);
  }
});

/* ============================================================
   POLICY GRID
   ============================================================ */

function policyActionFor(r, c, policyType) {
  if (isTerminal(r, c)) return null;
  if (GRID[r][c] === CELL.WALL) return null;

  if (policyType === 'optimal') {
    return mdp.policy ? mdp.policy[r][c] : null;
  }
  if (policyType === 'random') {
    return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
  }
  if (policyType === 'greedy') {
    const dr = 4 - r;
    const dc = 4 - c;
    if (Math.abs(dc) >= Math.abs(dr)) return dc >= 0 ? 'right' : 'left';
    return dr >= 0 ? 'down' : 'up';
  }
  return null;
}

function buildPolicyGrid(policyType) {
  const container = document.getElementById('policy-grid');
  container.innerHTML = '';

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement('div');
      cell.className = `grid-cell policy-cell ${cellClass(r, c)}`;

      const action = policyActionFor(r, c, policyType);
      const icon = cellIcon(r, c);

      if (icon) {
        cell.innerHTML = `<span class="cell-icon">${icon}</span><span class="pc-coord">(${r},${c})</span>`;
      } else if (action) {
        const arrowColor = policyType === 'optimal' ? 'var(--clr-cyan)' :
                           policyType === 'random'  ? 'var(--clr-amber)' : 'var(--clr-purple)';
        cell.innerHTML = `
          <span class="pc-arrow" style="color:${arrowColor}">${ACTION_LABEL[action]}</span>
          <span class="pc-coord">(${r},${c})</span>
        `;
      } else {
        cell.innerHTML = `<span class="pc-coord">(${r},${c})</span>`;
      }

      container.appendChild(cell);
    }
  }
}

// Policy selector buttons
document.querySelectorAll('.policy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.policy-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPolicyType = btn.dataset.policy;
    buildPolicyGrid(currentPolicyType);
    // If simulator is using this policy, reset
    if (!document.getElementById('manual-toggle').checked) {
      currentEpisode = new Episode(mdp, currentPolicyType);
      resetSimulator();
      document.getElementById('sim-mode-badge').textContent = `Policy: ${currentPolicyType.charAt(0).toUpperCase() + currentPolicyType.slice(1)}`;
    }
  });
});

/* ============================================================
   NAV PILL ACTIVE STATE (scroll-spy)
   ============================================================ */

const navPills = document.querySelectorAll('.nav-pill');

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      navPills.forEach(p => {
        p.classList.toggle('active', p.dataset.section === id);
      });
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.section').forEach(s => observer.observe(s));

/* ============================================================
   INIT
   ============================================================ */

function init() {
  buildStaticGrid();

  // Initialize value grid with boot-converged values
  buildValueGrid(mdp.V);
  document.getElementById('vi-iter').textContent = mdp.iteration;
  document.getElementById('vi-delta').textContent = '< 0.001';
  document.getElementById('vi-status').textContent = '✅ Converged';
  document.getElementById('vg-iter-label').textContent = mdp.iteration;
  document.getElementById('conv-iters').textContent = mdp.iteration;
  drawConvergenceChart();

  updateBellmanWorkedExample();
  populateValueTable();

  buildPolicyGrid('optimal');
  resetSimulator();
}

// Wait for fonts/layout to settle before drawing chart
window.addEventListener('load', init);
