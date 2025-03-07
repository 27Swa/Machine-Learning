# Machine Learning Tasks: Regression from Scratch

This repository explores two fundamental machine learning tasks implemented from scratch: **Linear Regression** and **Polynomial Regression**. The focus is on understanding the mathematical concepts and implementation steps without relying on external libraries. By the end of this project, you will have a solid grasp of how these algorithms work under the hood and how to implement them from the ground up.

---

## Task 1: Implementing Linear Regression from Scratch

### Overview
Linear regression is a supervised learning algorithm used for predicting continuous values. It assumes a linear relationship between the input feature(s) and the target variable. This task involves implementing linear regression from scratch, including the cost function and optimization process.

### Mathematical Formulation
The model is represented as: Y = θ₀ + θ₁X + ε

- **θ₀**: The intercept.
- **θ₁**: The weight of the feature.
- **ε**: The error term accounting for the difference between predicted and actual values.

### Implementation Steps
1. **Initialize Parameters:** Start with initial guesses for θ₀ and θ₁.
2. **Cost Function:** Use Mean Squared Error (MSE) to measure the model's performance. The MSE is defined as: MSE = (1 / 2m) * Σ(hθ(xⁱ) - yⁱ)²
where `hθ(x)` is the hypothesis function, `m` is the number of training examples, and `yⁱ` is the actual value.
3. **Optimization:** Employ Gradient Descent to iteratively adjust the parameters by minimizing the cost function. The update rules for θ₀ and θ₁ are:


θ₀ := θ₀ - α * (1 / m) * Σ(hθ(xⁱ) - yⁱ)
θ₁ := θ₁ - α * (1 / m) * Σ(hθ(xⁱ) - yⁱ) * xⁱ
where `α` is the learning rate.
4. **Training:** Update the parameters until the cost converges or a set number of iterations is reached.
5. **Prediction:** Use the optimized parameters to predict outcomes on new data.

---

## Task 2: Implementing Polynomial Regression from Scratch

### Overview
Polynomial regression extends linear regression by including polynomial terms of the features. This allows the model to capture non-linear relationships within the data. In this task, you will implement polynomial regression from scratch, with the added flexibility to choose the degree of the polynomial.

### Mathematical Formulation
For a polynomial of degree `d`, the model is given by: Y = θ₀ + θ₁X + θ₂X² + ... + θₙXⁿ + ε

- **θ₀**: The intercept.
- **θ₁, θ₂, ..., θₙ**: The weights corresponding to each power of `X`.
- **ε**: The error term.

### Implementation Steps
1. **Feature Transformation:** Augment the original feature `X` by adding polynomial features `(X², X³, ..., Xⁿ)`. This step allows the model to fit non-linear data.
2. **Model Training:** Use the linear regression framework on the transformed features. The hypothesis function becomes: hθ(x) = θ₀ + θ₁x + θ₂x² + ... + θₙxⁿ

3. **Optimization:** Adjust the parameters using an optimization technique (e.g., Gradient Descent) to minimize the cost function. The update rules for θⱼ (where `j = 0, 1, ..., n`) are:
θⱼ := θⱼ - α * (1 / m) * Σ(hθ(xⁱ) - yⁱ) * xⁱʲ
4. **Degree Selection:** Allow the user to choose the degree of the polynomial. Higher degrees can fit more complex patterns but may lead to overfitting. Experiment with different degrees to find the best fit.
5. **Evaluation:** Assess the model's performance using metrics like Mean Squared Error (MSE) or R-squared. Fine-tune the parameters and degree of the polynomial as needed.
