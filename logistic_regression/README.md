# Logistic Regression from First Principles

This module implements **logistic regression from scratch**, focusing on
the mathematical formulation, gradient-based optimization, and empirical
behavior during training.

---

## Objective

Given input features \(x \in \mathbb{R}^d\) and binary labels \(y \in \{0,1\}\),
logistic regression models:

\[
p(y=1 \mid x) = \sigma(w^\top x + b)
\]

where \(\sigma(\cdot)\) is the sigmoid function.

---

## Loss Function

Binary cross-entropy loss:

\[
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)
\right]
\]

Gradients are derived analytically and implemented explicitly.

---

## Implementation Details

- Fully vectorized NumPy implementation
- Manual gradient computation
- Optional L2 regularization
- Numerical stability considerations (log-sum tricks)

---

## Experiments

- Synthetic linearly separable data
- Effect of learning rate on convergence
- Comparison with `sklearn.linear_model.LogisticRegression`

---

## Observations

- Convergence highly sensitive to learning rate
- Gradient descent fails on poorly scaled features
- Saturation of sigmoid causes vanishing gradients

---

## Relation to Deep Learning

Logistic regression is equivalent to:
- a single-neuron neural network with sigmoid activation
- the binary classification head used in many deep models

Softmax regression generalizes this to multi-class classification and is
used as the output layer in transformer-based language models.

---

## Limitations

- Linear decision boundary
- Sensitive to feature scaling
- Not suitable for complex non-linear data

