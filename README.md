# Warehouse Robot MDP — RL Assignment 3

> **Interactive Finite MDP prototype** built with vanilla HTML/CSS/JavaScript.  
> Live demo → **[https://manukarthik2003.github.io/rl-assignment-3/](https://manukarthik2003.github.io/rl-assignment-3/)**

## Problem: Autonomous Warehouse Robot Navigation

A robot navigating a 5×5 warehouse grid to find the optimal path to a delivery zone, modelled as a Finite Markov Decision Process.

### MDP Components

| Component | Definition |
|-----------|-----------|
| **States** | 25 grid cells (r,c), r,c ∈ {0,1,2,3,4} |
| **Actions** | Up · Down · Left · Right |
| **Transitions** | Stochastic: 80% intended, 10% slip left, 10% slip right |
| **Rewards** | Goal +100 · Trap −50 · Step −1 · Dock +5 · Wall −10 |
| **Discount γ** | 0.9 |

### Features

- 📊 **Value Iteration** with animated convergence plot
- 🔔 **Bellman Equation** worked example with live substitution
- 🤖 **Live Simulator** — auto-run or manual keyboard control
- 🗺️ **Policy Map** — compare Optimal vs Random vs Greedy
- 📋 **Value Table** — all 25 states with Q-values

### Running Locally

Just open `index.html` in a browser — no build step needed!

```bash
# Or serve with any static server:
npx serve .
```

### Assignment Coverage

- ✅ Real-world sequential decision-making problem
- ✅ Agent, environment, states, actions, goals, rewards identified
- ✅ Finite MDP (S, A, P, R, γ) formally defined
- ✅ Episodes, returns, policy, value function defined
- ✅ State-transition diagrams and tables
- ✅ Bellman equation applied to state (2,2) with full worked example
- ✅ Policy influence on agent decisions explained and visualized
- ✅ Working interactive prototype/demo
