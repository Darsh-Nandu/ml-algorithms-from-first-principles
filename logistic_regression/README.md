# Logistic Regression from First Principles

This module implements **logistic regression from scratch**, focusing on
the mathematical objective, gradient-based optimization, and empirical
behavior during training.

The implementation avoids high-level ML libraries and makes all learning
dynamics explicit.

---

## Objective

Given input features  
x ∈ R^d  
and binary labels  
y ∈ {0, 1},

logistic regression models the probability:

p(y = 1 | x) = σ(wᵀx + b)

where:
- w is the weight vector
- b is the bias term
- σ(·) is the sigmoid function

---

## Sigmoid Function

The sigmoid function maps real-valued inputs to probabilities:

σ(z) = 1 / (1 + exp(-z))

This non-linearity allows linear models to perform binary classification.

---

## Loss Function

Binary cross-entropy loss is used:

L = -(1 / N) * Σ [ yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ) ]

where:
- N is the number of samples
- yᵢ is the true label
- ŷᵢ is the predicted probability

Gradients with respect to weights and bias are derived analytically and
implemented explicitly.

---

## Optimization

Parameters are optimized using **gradient descent**:

- weights and bias are updated iteratively
- learning rate controls convergence speed
- feature scaling significantly affects stability

No automatic differentiation is used.

---

## Implementation Details

- Fully vectorized NumPy implementation
- Explicit forward and backward passes
- Manual gradient computation
- Numerical stability considerations for log and exp operations
- Optional L2 regularization

---

## Experiments

Experiments are designed to study learning behavior, not just accuracy:

- Synthetic linearly separable datasets
- Effect of learning rate on convergence
- Impact of feature scaling
- Comparison with scikit-learn’s LogisticRegression

---

## Observations

- Convergence is highly sensitive to learning rate
- Poorly scaled features slow or destabilize training
- Sigmoid saturation leads to vanishing gradients
- Logistic regression fails on non-linearly separable data

---

## Relation to Deep Learning

Logistic regression is equivalent to:

- A single-neuron neural network with sigmoid activation
- The binary classification head used in many deep learning models

Softmax regression generalizes this idea to multi-class classification and
serves as the output layer in transformer-based language models.

---

## Limitations

- Linear decision boundary
- Sensitive to feature scaling
- Limited expressiveness compared to neural networks

---

## Takeaway

Logistic regression demonstrates how:
- probabilistic modeling
- optimization
- and representation learning

emerge from simple mathematical assumptions.

Understanding this model builds strong intuition for more complex
learning systems.
