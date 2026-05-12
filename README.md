# Neural Loss Reserving Engine
### Deep Learning Approaches to Non-Life Loss Reserving
A research and learning project exploring the application of neural network architectures to non-life insurance loss reserving, built by an actuarial student bridging classical reserving methods and modern deep learning.


---

## Overview

This project explores the application of neural network architectures to non-life insurance loss reserving. Starting from the familiar ground of Chain Ladder and Bornhuetter-Ferguson, each module progressively introduces more expressive models — making explicit the actuarial intuition behind every architectural choice.

The goal is both educational and experimental: to understand *why* and *how* deep learning methods can be applied to loss triangles, not just that they can.

---

## Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | **Neural Chain Ladder** | Reframe CL link ratios as a neural network. Show the equivalence with no hidden layers, then extend to deeper architectures. |
| 2 | **DeepTriangle** | LSTM-based model on CAS Schedule P data (Kuo 2019), predicting development tails per accident year. |
| 3 | **Probabilistic Output Heads** | Replace point predictions with distributional outputs (lognormal / negative binomial) for uncertainty quantification on ultimates. |
| 4 | **Transformer on Triangles** *(stretch)* | Self-attention over development periods as a replacement for the LSTM. |

Each module is accompanied by a Jupyter notebook that walks through the theory, implementation, and results.

---

## Data

**CAS Schedule P** — public loss triangles from US P&C insurers across multiple lines of business.

- Source: [Casualty Actuarial Society](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p)
- Format: Cumulative paid loss triangles, 10 accident years × 10 development years
- Lines included: Commercial Auto, Medical Malpractice, Other Liability, Private Passenger Auto, Workers' Compensation

---

## Project Structure

```
neural-reserving/
│
├── data/
│   └── cas_schedule_p/          # Raw and processed triangle data
│
├── src/
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # Model definitions (PyTorch)
│   ├── classical/               # Chain Ladder, BF implementations
│   └── utils/                   # Helpers, metrics, plotting
│
├── notebooks/
│   ├── 01_neural_chain_ladder.ipynb
│   ├── 02_deeptriangle.ipynb
│   ├── 03_probabilistic_heads.ipynb
│   └── 04_transformer.ipynb     # (stretch)
│
├── requirements.txt
└── README.md
```

---

## Stack

- **Python** 3.10+
- **PyTorch** — model definition and training
- **pandas / numpy** — data manipulation
- **matplotlib** — visualisation

---

## Background & Motivation

Classical reserving methods like Chain Ladder are elegant in their simplicity but assume fixed development patterns and offer no uncertainty quantification beyond bootstrapping. Neural networks offer a flexible alternative — but only become useful when their structure is grounded in actuarial reasoning.

This project is a learning vehicle for understanding recurrent and attention-based architectures, approached from an actuarial perspective. Prior background includes frequency-severity modelling, collective risk theory, Monte Carlo simulation, and basic neural network implementation (e.g. demonstrating that a NN without hidden layers is equivalent to OLS).

---

## References

- Kuo, K. (2019). *DeepTriangle: A Deep Learning Approach to Loss Reserving*. [arXiv:1804.09253](https://arxiv.org/abs/1804.09253)
- Mack, T. (1993). *Distribution-free calculation of the standard error of chain ladder reserve estimates.*
- England, P. & Verrall, R. (2002). *Stochastic Claims Reserving in General Insurance.*

---

## Status

🚧 Work in progress — modules being built sequentially as part of an active learning project.
