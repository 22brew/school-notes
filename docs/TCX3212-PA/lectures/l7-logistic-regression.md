---
title: "L7: Logistic Regression"
sidebar_label: "L7: Logistic Regression"
sidebar_position: 7
---

## Introduction to Classification

**Classification** is a supervised learning task where the response variable $Y$ takes values in a **finite, unordered set** (categories), rather than a continuous range.

Unlike regression (which predicts a quantity), classification predicts a **class label**.

*Example:* Given a customer's credit card balance and income, predict whether they will **default** (Yes/No).

### Why Not Use Linear Regression for Classification?

Linear regression can produce predictions outside $[0, 1]$, making it unsuitable as a probability estimate for binary outcomes. A value of $\hat{y} = 1.3$ or $\hat{y} = -0.2$ is meaningless as a probability. Threshold heuristics (e.g., predict Yes if $\hat{y} > 0.5$) are brittle and can require arbitrary threshold adjustments.

The solution: **Logistic Regression**, which models the *probability* of class membership.

---

## Logistic Regression

### The Logistic Function

Instead of modelling $Y$ directly, logistic regression models $P(Y = 1 \mid X) = p(X)$ — the probability that the outcome belongs to class 1.

The **logistic (sigmoid) function** maps any real-valued input to $[0, 1]$:

$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$

This S-shaped curve never goes below 0 or above 1, making it a valid probability model.

### Log-Odds (Logit) Transformation

To understand the model structure, rearrange:

$$\frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1 X} \qquad \text{(odds)}$$

$$\log\!\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X \qquad \text{(log-odds / logit)}$$

The **logit** (log-odds) is **linear** in $X$ — this is what makes logistic regression a "linear" model despite its non-linear output.

| Quantity | Formula | Range |
|----------|---------|-------|
| Probability $p(X)$ | $\frac{1}{1+e^{-(\beta_0+\beta_1 X)}}$ | $(0, 1)$ |
| Odds | $p(X) / (1-p(X))$ | $(0, \infty)$ |
| Log-odds (logit) | $\beta_0 + \beta_1 X$ | $(-\infty, \infty)$ |

### Making Predictions

Choose a **decision threshold** $\tau$:
- Predict $Y = 1$ if $p(X) > \tau$
- Common default: $\tau = 0.5$
- Lower threshold (e.g., $\tau = 0.1$) → more conservative (flag more potential defaults)

---

## Interpreting $\beta_1$

**In linear regression:** $\beta_1$ is the average change in $Y$ per unit increase in $X$.

**In logistic regression:**

- A one-unit increase in $X$ **adds $\beta_1$ to the log-odds**.
- The **odds** are multiplied by $e^{\beta_1}$.

$$\text{Odds change factor} = e^{\beta_1}$$

| Scenario | Interpretation |
|----------|---------------|
| $e^{\beta_1} > 1$ | Odds increase — positive association |
| $e^{\beta_1} < 1$ | Odds decrease — negative association |
| $e^{\beta_1} = 1$ | No change — no association |

**Percent change in odds:** $|e^{\beta_1} - 1| \times 100\%$

**Example — Default Dataset:**

$$\hat{\beta}_1 = 0.0055 \text{ (balance predictor)}$$

- A \$1 increase in balance increases the log-odds of default by 0.0055.
- The odds of default are multiplied by $e^{0.0055} \approx 1.0055$ per \$1 increase in balance — approximately a **0.55% increase** in odds.

**Predicted probability for balance = \$1,000:**

$$p = \frac{e^{-10.651 + 0.0055 \times 1000}}{1 + e^{-10.651 + 0.0055 \times 1000}} < 1\%$$

**For balance = \$2,000:** $p \approx 58.6\%$ — much higher risk.

---

## Estimating Coefficients: Maximum Likelihood

Unlike OLS (which minimises RSS), logistic regression uses **Maximum Likelihood Estimation (MLE)**:

For observation $i$, the likelihood of the outcome is:
- $p(x_i)$ if $y_i = 1$
- $1 - p(x_i)$ if $y_i = 0$

The **likelihood function:**

$$L(\beta_0, \beta_1) = \prod_{i: y_i=1} p(x_i) \cdot \prod_{i: y_i=0} (1 - p(x_i))$$

MLE chooses $\hat{\beta}_0, \hat{\beta}_1$ to **maximise** this likelihood — finding parameters that make the observed data most probable. In practice, numerical optimisation is used.

---

## Qualitative Predictors in Logistic Regression

Same approach as linear regression — create dummy variables:

*Example:* `student` variable (Yes/No) with reference = Non-student:

$$\hat{p}(\text{student}) = \frac{e^{\hat{\beta}_0 + \hat{\beta}_1 \cdot \mathbf{1}_{\text{student}}}}{1 + e^{\hat{\beta}_0 + \hat{\beta}_1 \cdot \mathbf{1}_{\text{student}}}}$$

Interpretation of $\hat{\beta}_1 = 0.4049$ (student = Yes):
- The odds of default for **students** are $e^{0.4049} \approx 1.499$ times the odds for non-students — approximately **49.9% higher odds**.

---

## Multiple Logistic Regression

Generalises to multiple predictors $X = (X_1, \ldots, X_p)$:

$$\log\!\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

**Default dataset with 3 predictors:**

$$\hat{p}(X) = \frac{e^{-10.869 + 0.0057 \cdot \text{balance} + 0.003 \cdot \text{income} - 0.6468 \cdot \text{student}}}{1 + e^{-10.869 + 0.0057 \cdot \text{balance} + 0.003 \cdot \text{income} - 0.6468 \cdot \text{student}}}$$

**Apparent contradiction:** In the single-predictor model, `student` has a *positive* coefficient (students have higher default probability). In the multiple model, it has a *negative* coefficient (for a given balance and income, students are *less* likely to default).

Why? This is **confounding**:
- Students tend to have **higher balances** — which increases default risk.
- But for the *same balance level*, students are actually **less risky** than non-students.
- The simple model's positive coefficient is driven by the fact that students carry more debt, not that being a student itself increases risk.

> **Practical implication:** A student is riskier than a non-student if you know nothing about their balance. But given the same balance, the student is less risky. Context and conditioning matter.

---

## Assessing Classification Accuracy

### Confusion Matrix

| | Predicted: No Default | Predicted: Default |
|--|---|---|
| **Actual: No Default** | True Negative (TN) | False Positive (FP) |
| **Actual: Default** | False Negative (FN) | True Positive (TP) |

**Example (10,000 customers):** TN = 9,627, FP = 40, FN = 228, TP = 105.

### Key Metrics

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **Accuracy** | $(TP + TN) / N$ | 97.3% | % correctly classified |
| **Sensitivity (Recall/TPR)** | $TP / (TP + FN)$ | 31.5% | % of actual positives correctly identified |
| **Specificity (TNR)** | $TN / (TN + FP)$ | 99.6% | % of actual negatives correctly identified |
| **Precision** | $TP / (TP + FP)$ | 72.4% | % of predicted positives that are true positives |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | — | Harmonic mean of precision and recall |
| **False Positive Rate (FPR)** | $FP / N_{\text{actual neg}}$ | 0.4% | Type I error; 1 − Specificity |

### Precision vs Recall Tradeoff

- **Precision** penalises false positives (predicting default when customer won't).
- **Recall (Sensitivity)** penalises false negatives (missing actual defaults).
- They are inversely related: lowering the threshold increases recall but decreases precision.

**Domain-specific considerations:**
- Medical testing for disease: prioritise **recall** (don't miss sick patients).
- Criminal sentencing: prioritise **precision** (don't wrongly convict innocent people).

**Rule of thumb:** Choose a threshold that keeps both precision and recall acceptably high for your application.

### ROC Curve and AUC

The **ROC (Receiver Operating Characteristic) curve** plots TPR (Sensitivity) vs FPR (1 − Specificity) across all possible thresholds.

- Top-left corner = ideal (high TPR, low FPR).
- Diagonal line = random classifier.
- **AUC (Area Under Curve):** Summarises overall performance across all thresholds. AUC = 1 is perfect; AUC = 0.5 is random.

---

## Practice Questions

**Q1.** For a binary classification task with real-valued inputs in $(-\infty, \infty)$, which function maps inputs to valid probabilities in $[0, 1]$?

- **A) ✓** $1/(1+e^{-x})$ — the sigmoid/logistic function
- B) $\log(x)$ — unbounded below, undefined for $x \leq 0$
- C) $e^x$ — unbounded above, always positive

> **Why A:** The sigmoid function is the only one listed that maps all real numbers to $(0, 1)$.

---

**Q2.** In logistic regression, what expression gives the **odds** of outcome $Y = 1$ given input $X$?

- A) $1/(1+e^{-(b_0+b_1 X)})$ — this is the probability.
- B) $b_0 + b_1 X$ — this is the log-odds.
- **C) ✓** $e^{b_0 + b_1 X}$

> **Why C:** Odds = $p/(1-p)$. Starting from the logistic equation: $p/(1-p) = e^{\beta_0 + \beta_1 X}$.

---

**Q3.** A logistic regression is fitted: $\text{logit}(p) = b_0 + 0.0055 \cdot \text{balance}$. Which is the correct interpretation of $b_1 = 0.0055$?

- A) Each additional \$1 in balance increases the probability of default by 0.55%.
- **B) ✓** For every additional \$1 in credit balance, the log-odds of default increase by 0.0055.
- C) For every additional \$1 in credit balance, the odds of default increase by exactly 0.0055.

> **Why B:** $b_1$ is the coefficient on the log-odds scale. The change in *probability* is not constant (it depends on current $p$); the change in *odds* is multiplicative (×$e^{0.0055}$), not additive.

---

**Q4.** A logistic regression predicts credit default with student as a predictor (reference = non-student). Student[Yes] coefficient $b_1 = 0.4049$. Which statement is correct?

- A) Probability of defaulting is 49.9% higher for students.
- B) Odds of default are ~4.049 times higher for students.
- **C) ✓** Odds of default for students are $e^{0.4049} \approx 1.499$ times the odds for non-students.

> **Why C:** The odds *ratio* is $e^{b_1}$. The *percent increase* in odds is $(e^{0.4049} - 1) \times 100\% \approx 49.9\%$. This is an increase in *odds*, not in *probability*.
