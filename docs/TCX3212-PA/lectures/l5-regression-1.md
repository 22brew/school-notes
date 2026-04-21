---
title: "L5: Regression I"
sidebar_label: "L5: Regression I"
sidebar_position: 5
---

## Introduction

Regression is a supervised learning technique where the response variable $Y$ is **quantitative** (continuous). The goal is to estimate the function $f$ such that $Y \approx f(X)$, enabling both prediction and inference.

---

## Simple Linear Regression

**Simple linear regression** assumes a single predictor $X$ and a linear relationship with response $Y$:

$$Y \approx \beta_0 + \beta_1 X$$

- $\beta_0$ — **intercept**: the expected value of $Y$ when $X = 0$.
- $\beta_1$ — **slope**: the average change in $Y$ for a one-unit increase in $X$.

Both $\beta_0$ and $\beta_1$ are unknown **population parameters** estimated from data.

### Motivating Example: Advertising

*Dataset:* `Advertising.csv` — 200 markets with TV ad spend (thousands \$) and Sales (thousands units).

Hypothesis: $\text{Sales} \approx \beta_0 + \beta_1 \cdot \text{TV}$

### Estimating Coefficients: Least Squares

We want to find $\hat{\beta}_0$ and $\hat{\beta}_1$ such that the fitted line $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ is "as close as possible" to the observed data.

**Residuals:**
$$e_i = y_i - \hat{y}_i$$

**Residual Sum of Squares (RSS):**
$$\text{RSS} = e_1^2 + e_2^2 + \cdots + e_n^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

The **Ordinary Least Squares (OLS)** estimators minimise RSS:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

### Interpreting Coefficients

For the Advertising example:

$$\widehat{\text{Sales}} = 7.033 + 0.04754 \cdot \text{TV}$$

- **Intercept (7.033):** When TV spending = \$0, average predicted sales = 7,033 units. (May have no practical meaning.)
- **Slope (0.04754):** Each additional \$1,000 in TV spending is associated with ~47.5 additional units sold on average.

**General interpretation:**
- $\hat{\beta}_0$ — predicted mean of $Y$ when all $X$s = 0.
- $\hat{\beta}_k$ — a one-unit increase in $X_k$ is associated with $|\hat{\beta}_k|$ unit increase (if positive) or decrease (if negative) in $Y$, *holding all other variables constant*.

---

## Measuring Model Performance

### $R^2$ (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}, \qquad \text{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2$$

- $R^2 \in [0, 1]$: proportion of variance in $Y$ explained by $X$.
- $R^2 = 0$: the model explains none of the variability (no better than the mean).
- $R^2 = 1$: the model perfectly explains the variability.

> ⚠️ $R^2$ always increases as more predictors are added — even irrelevant ones. Use **Adjusted $R^2$** for model comparison.

**Adjusted $R^2$** penalises for adding unnecessary predictors:

$$\bar{R}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where $p$ is the number of predictors. Adjusted $R^2 \leq R^2$ always.

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Measures the average squared prediction error. More directly interpretable than $R^2$ when prediction accuracy matters.

### F-Statistic

Tests whether **any** predictor is useful in the model (relevant in multiple regression):

- $H_0$: $\beta_1 = \beta_2 = \cdots = \beta_k = 0$ (no predictor is useful)
- $H_1$: at least one $\beta_j \neq 0$

A large F-statistic (and small p-value < 0.05) means we reject $H_0$ — the model as a whole is significant.

---

## Multiple Linear Regression

In practice, we have multiple predictors. Instead of running separate simple regressions (which ignores relationships among predictors), we use **multiple linear regression**:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon$$

For the Advertising dataset:

$$\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \beta_3 \cdot \text{Newspaper} + \varepsilon$$

Coefficients are again estimated by minimising RSS.

### Why Not Run Separate Simple Regressions?

- Cannot make a *single* prediction of $Y$ that accounts for all predictors simultaneously.
- Each isolated regression ignores correlations among predictors, leading to unexpected and often misleading results.

### Example: Conflicting Results for Newspaper

When Newspaper is regressed alone on Sales, it appears significant. But in the full multiple regression model, **Newspaper is not significant**.

Why? Because **Radio and Newspaper spending are positively correlated** (markets that spend more on radio tend to spend more on newspaper too). Newspaper is acting as a *surrogate* for Radio — it gets statistical "credit" that really belongs to Radio.

```python
# Correlation matrix reveals this
correlation_matrix = advertising_data.corr()
```

This illustrates **confounding** — a predictor may appear important in isolation but be insignificant once other correlated predictors are included.

### R² in Multiple Regression

Adding more predictors always increases $R^2$, even if they are useless:

| Predictors | $R^2$ |
|-----------|--------|
| TV only | 0.612 |
| TV + Radio | 0.897 |
| TV + Radio + Newspaper | 0.897 (tiny increase) |

The near-zero improvement from adding Newspaper confirms it is not contributing useful information beyond what Radio already captures. For model selection, use **Adjusted $R^2$**, **AIC**, or **BIC**.

---

## Assumptions of Linear Regression

Linear regression relies on four assumptions for valid inference (p-values and confidence intervals):

| Assumption | Description |
|------------|-------------|
| **Linearity** | The true relationship between $Y$ and $X$ is linear in the parameters. |
| **Normal errors** | The error terms $\varepsilon_i$ are normally distributed. |
| **Homoscedasticity** | The variance of errors is constant across all values of $X$ (constant variance). |
| **Independence** | Errors are independent of each other. |

When these hold, OLS is the **Best Linear Unbiased Estimator (BLUE)** — the Gauss-Markov theorem.

---

## Problems with Linear Regression

The following issues can violate assumptions or degrade model performance:

### Non-Linearity
The true relationship is curved. **Diagnostic:** residual plot (residuals vs fitted values) shows a systematic pattern rather than random scatter around zero. **Fix:** add polynomial terms or transform predictors.

### Correlated Errors
Residuals are correlated over time/space — common in time series data. **Diagnostic:** Durbin-Watson test; residuals vs time plot shows alternating positive/negative clusters.

### Non-Constant Variance (Heteroscedasticity)
Variance of residuals increases (or decreases) with $\hat{y}$. **Diagnostic:** residual plot fans out. **Fix:** log-transform $Y$ or use weighted least squares.

### Outliers
Points where $Y$ is far from the predicted value. Outliers inflate RSS and distort coefficient estimates. **Diagnostic:** standardised residuals beyond ±3.

### High-Leverage Points
Points where $X$ is far from the usual range of $X$. These can have a disproportionate influence on the fitted line even if their $Y$ is well-predicted. **Diagnostic:** leverage statistic (hat values).

### Collinearity

Two or more predictors are highly correlated with each other. This makes it difficult to determine which predictor is truly driving the response.

**Effects of collinearity:**
- Standard errors of coefficients inflate → confidence intervals widen → harder to detect significance.
- Coefficient estimates become unstable (sensitive to small changes in the data).

**Diagnostic:** Variance Inflation Factor (VIF). $\text{VIF} > 10$ signals serious collinearity.

**Fix:** Remove one of the correlated predictors, combine them (e.g., average), or use ridge regression (covered in L6).

---

## Python: Fitting Linear Regression

```python
import statsmodels.formula.api as smf
import pandas as pd

# Load data
advertising = pd.read_csv('Advertising.csv')

# Simple linear regression: Sales ~ TV
model = smf.ols('Sales ~ TV', data=advertising).fit()
print(model.summary())

# Multiple linear regression: Sales ~ TV + Radio + Newspaper
model_multi = smf.ols('Sales ~ TV + Radio + Newspaper', data=advertising).fit()
print(model_multi.summary())
```

Key outputs from `model.summary()`:
- **coef**: estimated coefficients $\hat{\beta}$
- **std err**: standard errors of coefficients
- **t / P>|t|**: t-statistic and p-value for each coefficient
- **[0.025, 0.975]**: 95% confidence interval for each coefficient
- **R-squared / Adj. R-squared**: model fit
- **F-statistic / Prob (F-statistic)**: overall model significance

---

## Practice Questions

**Q1.** In the regression `Sales ~ TV + Radio + Newspaper`, the multiple regression shows no significant relationship between Newspaper and Sales, while a simple regression of `Sales ~ Newspaper` does show a significant relationship. Does this make sense?

> **Answer:** Yes — this is a classic **confounding / omitted variable** problem. Newspaper spending is correlated with Radio spending (markets that spend more on radio also tend to spend more on newspaper). In the simple regression, Newspaper gets credit for the effect of Radio. Once Radio is controlled for in the multiple regression, the Newspaper effect disappears. The multiple regression result is more trustworthy.

**Q2.** What does the fitted equation $\widehat{\text{Sales}} = 7.033 + 0.04754 \cdot \text{TV}$ tell us?

> **Answer:** For every additional \$1,000 spent on TV advertising, average Sales increase by approximately 47.5 units (0.04754 × 1000). The intercept 7,033 is the predicted Sales when TV spending is zero — may not be practically meaningful.
