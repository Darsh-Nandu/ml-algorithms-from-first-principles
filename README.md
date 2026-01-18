# ML Algorithms from First Principles

This repository contains implementations of core machine learning algorithms **from scratch**, with minimal reliance on high-level ML libraries.

The goal is not performance or production readiness, but **deep understanding** — implementing algorithms from their mathematical foundations and analyzing their behavior through experiments.

---

## Motivation

Modern ML frameworks abstract away critical details such as:
- optimization dynamics  
- gradient flow  
- numerical stability  
- inductive biases  

By implementing algorithms from first principles, this repository aims to:
- build strong intuition about learning dynamics
- bridge classical ML with modern deep learning
- understand what libraries like PyTorch and scikit-learn automate

This work complements my research-oriented interests in **representation learning, LLMs, and continual pretraining**.

---

## Design Philosophy

- **Minimal dependencies**: primarily NumPy
- **Explicit mathematics**: loss functions and gradients are written manually
- **Readable code** over clever tricks
- **Experiment-driven**: every algorithm is tested on data
- **Failure-aware**: limitations and breakdowns are documented

---

## Repository Structure

Each algorithm is implemented as a self-contained module:

