# AIML Coursework

## Overview
During the first semester of my Computer Science (Conversion) Artificial Intelligence and Machine Learning module, I completed two coding assignments. The first assignment involved implementing a simple two-layer convolutional neural network, while the second focused on using a dynamic programming recurrence to compute minimum sum combinations.

## Technology Stack
**Language:** Python

## Implementation

### Convolutional Neural Network

- The `dual` class is defined to represent a dual number type consisting of a value and the value of its derivative. This format is useful in machine learning, as derivatives play a critical role in a model’s error function, which is then used to improve model accuracy. This setup enables the implementation of automatic differentiation, propagating derivative values through the neural network. The methods within the `dual` class reflect the application of the chain rule for differentiation.

- `builtins` is used to ensure that the `max` method is only overridden when the input value is of type `dual`.

### Minimum Sum Combinations

- The `minsumcomb` function takes an array `x` and a result combination length `M`. The function returns the combination of `M` elements in `x` that yields the minimum sum. This is achieved using a Bellman recursion, whereby the optimal solution from the previous iteration is reused in the current one. This results in a performant and efficient solution.
 



