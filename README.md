# Anesthesia RL Repository

## Overview

This repository implements reinforcement learning algorithms for automated anesthesia control. The models control propofol infusion rates to maintain target BIS (Bispectral Index) levels during surgery using PK/PD (Schnider model) simulations.

---

## State Representation

All algorithms share the same fuzzy state representation:

```
State = (BIS_ERROR, DELTA_BIS)
  ├── BIS_ERROR = BIS - BIS_TARGET    (range: -50 to +50)
  └── DELTA_BIS = BIS[t] - BIS[t-1]   (range: -30 to +30)
```

Fuzzification: 3 membership functions × 2 values × 10 bins = 1,000,000 states.

## Actions

```
ACTIONS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]  ml/min
```

---

## 1. Q-Learning (Tabular)

A simple tabular RL approach using a Q-table to learn optimal propofol infusion policies.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Q-LEARNING ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐      ┌──────────────┐      ┌───────────────┐       │
│   │  BIS[t]  │─────▶│   Compute    │─────▶│  Fuzzy State  │       │
│   │          │      │ BIS_ERROR    │      │   S = (6 features)    │
│   └──────────┘      │ DELTA_BIS    │      └───────┬───────┘       │
│                    └──────────────┘              │               │
│                                                  ▼               │
│                    ┌──────────────────────────────────────┐     │
│                    │           Q-TABLE (1M × 7)           │     │
│                    │    Q[s, a] = expected future reward  │     │
│                    └──────────────────┬───────────────────┘     │
│                                       │                          │
│                    ┌──────────────────▼───────────────────┐     │
│                    │     ε-greedy Action Selection     │     │
│                    │  a = argmax(Q[s,:])  (1-ε prob)   │     │
│                    │  a = random          (ε prob)     │     │
│                    └──────────────────┬───────────────────┘     │
│                                       │                          │
│                                       ▼                          │
│   ┌──────────┐      ┌──────────────────────────────────────┐     │
│   │  Reward  │◀────│  r = -|BIS_ERROR| - 0.5×|DELTA_BIS|  │     │
│   └──────────┘      └──────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hyperparameters**: EPISODES=1,000,000 | STEPS=120 | ALPHA=0.2 | GAMMA=0.69 | EPSILON=0.1

**Status**: Needs more training — current performance shows insufficient propofol delivery.

---

## 2. DP-Policy (Policy Iteration)

Tabular Q-learning with iterative policy improvement. Learns Q-values through experience, then extracts the optimal policy.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DP-POLICY ITERATION ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │                    TRAINING LOOP (10,000 episodes)           ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │                                                                  ││
│   │   For each episode:                                            ││
│   │   ┌──────────┐      ┌──────────────┐      ┌───────────────┐   ││
│   │   │  BIS[t]  │─────▶│   Compute    │─────▶│  Fuzzy State  │   ││
│   │   └──────────┘      │ BIS_ERROR    │      │      S        │   ││
│   │                    │ DELTA_BIS    │      └───────┬───────┘   ││
│   │                    └──────────────┘              │           ││
│   │                                                  ▼           ││
│   │   ┌──────────────────────────────────────────────────────┐  ││
│   │   │           Q-Table Update (SARS')                      │  ││
│   │   │  Q[s,a] ← Q[s,a] + α[r + γ·max Q[s',:] - Q[s,a]]     │  ││
│   │   └──────────────────────────────────────────────────────┘  ││
│   │                                                                  ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │                    POLICY EXTRACTION                            ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │   π(s) = argmax_a Q(s,a)    (with ε-greedy exploration)       ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hyperparameters**: EPISODES=10,000 | STEPS=120 | ALPHA=0.2 | GAMMA=0.69 | EPSILON=0.1

**Status**: Needs more training — high wobble indicates unstable control.

---

## 3. DP-Value (Value Iteration)

Dynamic programming approach that builds a transition model and uses Bellman equations to compute optimal values iteratively.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DP-VALUE ITERATION ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │              TRANSITION MATRIX COMPUTATION                     ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │   For each state s and action a:                               ││
│   │     s' = T(s, a)  (deterministic: BIS_ERROR decreases)         ││
│   │     R(s, a) = -|BIS_ERROR'| - 0.5×|DELTA_BIS'|                 ││
│   │   P[s, a] = s'  (next state index)                            ││
│   │   R[s, a] = reward                                             ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │                 VALUE ITERATION LOOP                           ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │                                                                  ││
│   │   Repeat until convergence (1,000,000 iterations):            ││
│   │   ┌─────────────────────────────────────────────────────────┐ ││
│   │   │   Q[s,a] = R[s,a] + γ × V[P[s,a]]   (Bellman equation)  │ ││
│   │   │   V[s] = max_a Q[s,a]                                   │ ││
│   │   └─────────────────────────────────────────────────────────┘ ││
│   │                                                                  ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │                    POLICY EXTRACTION                            ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │   π(s) = argmax_a (R[s,a] + γ × V[P[s,a]])                    ││
│   │         (with ε-greedy: 10% random, 90% optimal)             ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hyperparameters**: ITERATIONS=1,000,000 | GAMMA=0.69 | EPSILON=0.1

**Status**: Needs more training — same policy as DP-Policy.

---

## 4. DQN (Deep Q-Network)

Deep RL approach using a neural network to approximate Q-values, with experience replay and target network for stable training.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DQN ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │                    NEURAL NETWORK                               ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │                                                                  ││
│   │   Input: Fuzzy State (6 features)                               ││
│   │       ↓                                                         ││
│   │   FC Layer 1: 256 neurons, ReLU                                 ││
│   │       ↓                                                         ││
│   │   FC Layer 2: 128 neurons, ReLU                                 ││
│   │       ↓                                                         ││
│   │   Output: Q-values for 7 actions (linear)                       ││
│   │                                                                  ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│   ┌────────────────────────────────────────────────────────────────┐│
│   │                    TRAINING LOOP                                ││
│   ├────────────────────────────────────────────────────────────────┤│
│   │                                                                  ││
│   │   For each episode:                                            ││
│   │   ┌──────────┐      ┌──────────────┐      ┌───────────────┐   ││
│   │   │  BIS[t]  │─────▶│   Compute    │─────▶│  Fuzzy State  │   ││
│   │   └──────────┘      │ BIS_ERROR    │      │      S        │   ││
│   │                    │ DELTA_BIS    │      └───────┬───────┘   ││
│   │                    └──────────────┘              │           ││
│   │                                                  ▼           ││
│   │   ┌──────────────────────────────────────────────────────┐   ││
│   │   │  ε-greedy: Random if rand()<ε else Neural Net      │   ││
│   │   └──────────────────┬───────────────────────────────┬────┘   ││
│   │                      │                              │        ││
│   │                      ▼                              ▼        ││
│   │   ┌──────────┐  ┌──────────────┐      ┌──────────────────┐  ││
│   │   │  Action  │  │ Store (S,A,R,S')│      │  Target Q      │  ││
│   │   │  (u)     │  │ in Replay Buffer│◀─────│  = R + γ·max Q'│  ││
│   │   └──────────┘  └──────────────┘      └────────┬─────────┘  ││
│   │                                                  │           ││
│   │                                                  ▼           ││
│   │   ┌──────────────────────────────────────────────────────┐  ││
│   │   │     Gradient Descent: Loss = (Q - Target)²           │  ││
│   │   └──────────────────────────────────────────────────────┘  ││
│   │                                                                  ││
│   └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Architecture**: Input(6) → FC(256, ReLU) → FC(128, ReLU) → Output(7)

**Hyperparameters**: EPISODES=10,000 | STEPS=120 | LR=1e-3 | GAMMA=0.69 | EPSILON=0.3→0.01 | BUFFER=10,000 | BATCH=32

**Status**: Needs more training — similar performance to DP models.

---

## PK/PD Model (Schnider)

### Compartmental Parameters
| Parameter | Value | Description |
|------------|-------|-------------|
| V1 | 4.27 L | Central compartment |
| V2 | 18.9 L | Shallow peripheral |
| V3 | 238.0 L | Deep peripheral |
| k10 | 0.38 | Elimination rate |
| k12/k21 | 0.30/0.20 | Shallow exchange |
| k13/k31 | 0.19/0.0035 | Deep exchange |
| ke0 | 0.17 | Effect site equilibration |

### BIS Model
| Parameter | Value |
|-----------|-------|
| BIS_0 | 95.0 |
| BIS_MAX | 75.0 |
| EC50 | 3.5 μg/mL |
| HILL | 2.5 |
| BIS_TARGET | 50.0 |

---

*Repository by Mohamed Haytam Soukraty*
