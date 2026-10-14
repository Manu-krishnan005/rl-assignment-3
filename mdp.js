/**
 * mdp.js — Warehouse Robot MDP Core Engine
 * Finite MDP: 5x5 Grid, stochastic transitions, Value Iteration
 */

'use strict';

/* ============================================================
   GRID / ENVIRONMENT DEFINITION
   ============================================================ */

const ROWS = 5;
const COLS = 5;
const GAMMA = 0.9;
const EPSILON = 0.001; // convergence threshold

// Cell types
const CELL = {
  OPEN:  'open',
  WALL:  'wall',
  TRAP:  'trap',
  GOAL:  'goal',
  START: 'start',
  DOCK:  'dock',
};

// Grid layout [row][col] → cell type
// Coordinates: row 0 = top, row 4 = bottom
const GRID = [
  //  c0      c1      c2      c3      c4
  [CELL.START, CELL.OPEN, CELL.OPEN, CELL.OPEN, CELL.DOCK],   // row 0
  [CELL.OPEN,  CELL.WALL, CELL.TRAP, CELL.OPEN, CELL.OPEN],   // row 1
  [CELL.OPEN,  CELL.OPEN, CELL.OPEN, CELL.WALL, CELL.OPEN],   // row 2
  [CELL.OPEN,  CELL.WALL, CELL.OPEN, CELL.OPEN, CELL.OPEN],   // row 3
  [CELL.OPEN,  CELL.OPEN, CELL.OPEN, CELL.OPEN, CELL.GOAL],   // row 4
];

// Rewards per cell type (reward for entering that cell)
const REWARD = {
  [CELL.OPEN]:  -1,
  [CELL.WALL]:  -10,   // bounced back = stay in place but penalised
  [CELL.TRAP]:  -50,
  [CELL.GOAL]:  +100,
  [CELL.START]: -1,    // treated as open during movement
  [CELL.DOCK]:  +5,
};

// Terminal states
function isTerminal(r, c) {
  return GRID[r][c] === CELL.GOAL || GRID[r][c] === CELL.TRAP;
}

/* ============================================================
   ACTIONS & TRANSITIONS
   ============================================================ */

const ACTIONS = ['up', 'down', 'left', 'right'];
const ACTION_LABEL = { up: '↑', down: '↓', left: '←', right: '→' };
const ACTION_EMOJI = { up: '⬆', down: '⬇', left: '⬅', right: '➡' };

const DELTA = {
  up:    [-1,  0],
  down:  [ 1,  0],
  left:  [ 0, -1],
  right: [ 0,  1],
};

// Perpendicular actions for slipping
const PERP = {
  up:    ['left', 'right'],
  down:  ['right', 'left'],
  left:  ['down', 'up'],
  right: ['up', 'down'],
};

// Try to move; if out of bounds or wall → stay in place
function tryMove(r, c, action) {
  const [dr, dc] = DELTA[action];
  const nr = r + dr;
  const nc = c + dc;
  if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) {
    return [r, c]; // bounce off border
  }
  if (GRID[nr][nc] === CELL.WALL) {
    return [r, c]; // bounce off wall
  }
  return [nr, nc];
}

/**
 * Get all (nextState, probability, reward) for a (state, action) pair.
 * Stochastic: 80% intended, 10% each perpendicular.
 */
function getTransitions(r, c, action) {
  if (isTerminal(r, c)) return []; // no transitions from terminal

  const transitions = [];

  const addTrans = (prob, act) => {
    const [nr, nc] = tryMove(r, c, act);
    const bounced = (nr === r && nc === c);
    let reward;
    if (bounced) {
      reward = REWARD[CELL.WALL]; // bounced
    } else {
      reward = REWARD[GRID[nr][nc]];
    }
    transitions.push({ nr, nc, prob, reward });
  };

  addTrans(0.80, action);
  const [p1, p2] = PERP[action];
  addTrans(0.10, p1);
  addTrans(0.10, p2);

  // Merge identical next states
  const merged = {};
  for (const t of transitions) {
    const key = `${t.nr},${t.nc}`;
    if (merged[key]) {
      merged[key].prob += t.prob;
    } else {
      merged[key] = { ...t };
    }
  }
  return Object.values(merged);
}

/* ============================================================
   VALUE ITERATION
   ============================================================ */

class MDP {
  constructor() {
    this.V = this._initValues();
    this.iteration = 0;
    this.converged = false;
    this.deltaHistory = [];
    this.policy = null;
    this.Qvals = null;
  }

  _initValues() {
    const V = [];
    for (let r = 0; r < ROWS; r++) {
      V.push([]);
      for (let c = 0; c < COLS; c++) {
        if (GRID[r][c] === CELL.GOAL) V[r].push(100);
        else if (GRID[r][c] === CELL.TRAP) V[r].push(-50);
        else if (GRID[r][c] === CELL.WALL) V[r].push(0);
        else V[r].push(0);
      }
    }
    return V;
  }

  /**
   * Compute Q(s,a) = sum over s' of P(s'|s,a) * [R(s,a,s') + gamma*V(s')]
   */
  computeQ(r, c, action) {
    const transitions = getTransitions(r, c, action);
    if (transitions.length === 0) {
      return GRID[r][c] === CELL.GOAL ? 100 : -50;
    }
    let q = 0;
    for (const t of transitions) {
      q += t.prob * (t.reward + GAMMA * this.V[t.nr][t.nc]);
    }
    return q;
  }

  /** One full sweep of Value Iteration. Returns max delta. */
  step() {
    if (this.converged) return 0;

    const newV = [];
    let maxDelta = 0;

    for (let r = 0; r < ROWS; r++) {
      newV.push([]);
      for (let c = 0; c < COLS; c++) {
        const type = GRID[r][c];
        if (type === CELL.GOAL) { newV[r].push(100); continue; }
        if (type === CELL.TRAP) { newV[r].push(-50); continue; }
        if (type === CELL.WALL) { newV[r].push(0);   continue; }

        let maxQ = -Infinity;
        for (const a of ACTIONS) {
          const q = this.computeQ(r, c, a);
          if (q > maxQ) maxQ = q;
        }
        newV[r].push(maxQ);
        maxDelta = Math.max(maxDelta, Math.abs(maxQ - this.V[r][c]));
      }
    }

    this.V = newV;
    this.iteration++;
    this.deltaHistory.push(maxDelta);

    if (maxDelta < EPSILON) {
      this.converged = true;
      this._extractPolicy();
    }

    return maxDelta;
  }

  /** Extract greedy policy from V */
  _extractPolicy() {
    this.policy = [];
    this.Qvals = [];
    for (let r = 0; r < ROWS; r++) {
      this.policy.push([]);
      this.Qvals.push([]);
      for (let c = 0; c < COLS; c++) {
        if (isTerminal(r, c) || GRID[r][c] === CELL.WALL) {
          this.policy[r].push(null);
          this.Qvals[r].push({ up: null, down: null, left: null, right: null });
          continue;
        }
        let bestA = null;
        let bestQ = -Infinity;
        const qMap = {};
        for (const a of ACTIONS) {
          const q = this.computeQ(r, c, a);
          qMap[a] = q;
          if (q > bestQ) { bestQ = q; bestA = a; }
        }
        this.policy[r].push(bestA);
        this.Qvals[r].push(qMap);
      }
    }
  }

  /** Run until convergence. Returns number of iterations. */
  runToConvergence() {
    let safety = 0;
    while (!this.converged && safety < 5000) {
      this.step();
      safety++;
    }
    if (!this.policy) this._extractPolicy();
    return this.iteration;
  }

  reset() {
    this.V = this._initValues();
    this.iteration = 0;
    this.converged = false;
    this.deltaHistory = [];
    this.policy = null;
    this.Qvals = null;
  }

  /** Get best action using value function (optimal policy). */
  bestAction(r, c) {
    if (isTerminal(r, c) || GRID[r][c] === CELL.WALL) return null;
    if (this.policy) return this.policy[r][c];
    let bestA = null;
    let bestQ = -Infinity;
    for (const a of ACTIONS) {
      const q = this.computeQ(r, c, a);
      if (q > bestQ) { bestQ = q; bestA = a; }
    }
    return bestA;
  }
}

/* ============================================================
   EPISODE SIMULATION
   ============================================================ */

class Episode {
  constructor(mdp, policy = 'optimal') {
    this.mdp = mdp;
    this.policyType = policy;
    this.r = 0;
    this.c = 0;
    this.step = 0;
    this.totalReturn = 0;
    this.discount = 1;
    this.done = false;
    this.outcome = null;
    this.history = []; // [{r,c,action,reward,nr,nc}]
  }

  /** Choose action based on policy type */
  chooseAction(r, c) {
    if (this.policyType === 'optimal') {
      return this.mdp.bestAction(r, c) || 'right';
    }
    if (this.policyType === 'random') {
      return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
    }
    if (this.policyType === 'greedy') {
      // Move toward goal (4,4), no lookahead
      const dr = 4 - r;
      const dc = 4 - c;
      if (Math.abs(dc) >= Math.abs(dr)) {
        return dc >= 0 ? 'right' : 'left';
      } else {
        return dr >= 0 ? 'down' : 'up';
      }
    }
    return 'right';
  }

  /** Stochastic action execution */
  executeAction(r, c, action) {
    const rand = Math.random();
    let actualAction;
    if (rand < 0.80) {
      actualAction = action;
    } else if (rand < 0.90) {
      actualAction = PERP[action][0];
    } else {
      actualAction = PERP[action][1];
    }
    const [nr, nc] = tryMove(r, c, actualAction);
    const bounced = (nr === r && nc === c);
    let reward;
    if (bounced) {
      reward = REWARD[CELL.WALL];
    } else {
      reward = REWARD[GRID[nr][nc]];
    }
    return { nr, nc, reward, actualAction, intended: actualAction === action };
  }

  /** Take one step. Returns step info. */
  takeStep(manualAction = null) {
    if (this.done || this.step >= 200) {
      this.done = true;
      this.outcome = 'timeout';
      return null;
    }

    const action = manualAction || this.chooseAction(this.r, this.c);
    const { nr, nc, reward, actualAction, intended } = this.executeAction(this.r, this.c, action);

    this.totalReturn += this.discount * reward;
    this.discount *= GAMMA;

    const stepInfo = {
      r: this.r, c: this.c,
      action, actualAction, intended,
      nr, nc, reward,
      step: this.step + 1,
      totalReturn: this.totalReturn,
      vCurrent: this.mdp.V[this.r][this.c],
    };

    this.history.push(stepInfo);
    this.r = nr;
    this.c = nc;
    this.step++;

    if (GRID[nr][nc] === CELL.GOAL) {
      this.done = true;
      this.outcome = 'success';
    } else if (GRID[nr][nc] === CELL.TRAP) {
      this.done = true;
      this.outcome = 'failure';
    } else if (this.step >= 200) {
      this.done = true;
      this.outcome = 'timeout';
    }

    return stepInfo;
  }
}

/* ============================================================
   EXPORTS (global)
   ============================================================ */

window.MDP_ENGINE = {
  ROWS, COLS, GAMMA, EPSILON,
  CELL, GRID, REWARD, ACTIONS, ACTION_LABEL, ACTION_EMOJI,
  isTerminal, getTransitions, tryMove,
  MDP, Episode,
};
