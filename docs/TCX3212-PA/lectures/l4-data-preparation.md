---
title: "L4: Data Preparation"
sidebar_label: "L4: Data Preparation"
sidebar_position: 4
---

## Overview

Data preparation (also called **data pre-processing**) corresponds to Step 3 of CRISP-DM. It is typically the **most time-consuming** phase of any analytics project and has the largest impact on model accuracy. Real-world data is almost never clean and ready for modelling.

The four stages of data preparation are:

1. **Data Cleaning** — handle missing values, noise, and inconsistencies.
2. **Data Transformation** — normalise, discretise, or construct new features.
3. **Data Reduction** — reduce the number of variables or records.
4. **Resampling Methods** — techniques for model evaluation when test data is not separately available.

---

## Data Anomalies

Mathematical models can only produce reliable results when the underlying data is of high quality. Common anomalies include:

| Anomaly | Description |
|---------|-------------|
| **Incompleteness** | Missing values in one or more attributes |
| **Noise** | Erroneous or extreme values (outliers) |
| **Inconsistency** | Contradictions between records or fields |

---

## Data Cleaning

### Handling Missing Values

Missing values appear as `NaN`, `None`, empty strings `""`, or suspicious placeholders like `0` in contexts where zero is physically impossible.

```python
df.isnull().sum()   # Count missing values per column
```

**Types of missingness:**

| Type | Description |
|------|-------------|
| **MCAR** – Missing Completely at Random | Missingness does not depend on observed or unobserved values. Each observation has an equal chance of being missing. |
| **MAR** – Missing at Random | Missingness is related to *observed* values (but not to the missing value itself). |
| **MNAR** – Missing Not at Random | Missingness is related to the *unobserved* (missing) value — the hardest case. |

**Strategies for handling missing data:**

1. **Elimination** — drop rows with missing values, or drop the column entirely.  
   *Risk:* smaller dataset; potential bias if data is not MCAR.

2. **Inspection** — domain experts recommend plausible substitute values.  
   *Risk:* subjective; may introduce bias.

3. **Identification** — replace missing values with a sentinel value outside the normal range (e.g., `-1` for a positive-only variable; a new `NOT_FOUND` category for categoricals).

4. **Substitution (Imputation):**
   - Replace with column **mean** (numerical) or **mode** (categorical).
   - **Time-series interpolation:** use nearby observations.
   - **Model-based imputation:** predict missing values using a regression/classification model on the other features.
   - **Hot-deck imputation:** randomly sample from similar records in the same subgroup.

> ⚠️ Imputation choice affects both bias and uncertainty quantification — there is substantial statistical literature on this. For high-stakes applications, multiple imputation is preferred.

### Detecting and Handling Noise (Outliers)

Noisy data may result from malfunctioning sensors, data entry errors, or inaccurate conversions.

**Detection using the Interquartile Range (IQR):**

$$\text{IQR} = Q_3 - Q_1$$

| Threshold | Category |
|-----------|----------|
| $< Q_1 - 1.5 \cdot \text{IQR}$ or $> Q_3 + 1.5 \cdot \text{IQR}$ | Outlier |
| $< Q_1 - 3 \cdot \text{IQR}$ or $> Q_3 + 3 \cdot \text{IQR}$ | Extreme value |

**Example:**  
Data: `[45, 450, 20, 69, 9, 66, 11, 42, 9, 126, 47, 43, 24, 94, 89, 16, 83, 59, 57, 273, 70, 45, 40]`

- Median = 47, Q1 = 32, Q3 = 76.5, IQR = 44.5
- Outlier threshold: above Q3 + 1.5·IQR = 143.25
- Values 273 and 450 are outliers.

The box plot visualises this directly — outliers appear as individual points beyond the whiskers.

**Handling:** correct the value (if the true value is known), or remove the record if the noise is irreparable.

---

## Data Transformation

### Normalisation

Normalisation rescales variables so that features with different units or magnitudes are comparable. This is especially important for distance-based and regularised models.

**1. Decimal Scaling**

Shift the decimal point left by $h$ positions so values fall in $[-1, +1]$:

$$x' = x / 10^h$$

Example: `123, -456, 23, -42` with $h = 3$ → `0.123, -0.456, 0.023, -0.042`

**2. Log Scaling**

Apply a logarithm to compress wide-ranging data. Since $\log(0)$ is undefined, add a constant to shift all values above 0 first:

$$x' = \log(x + c)$$

**3. Min-Max Normalisation**

Rescale values to a target range $[x'_{\min}, x'_{\max}]$ (commonly $[0, 1]$ or $[-1, 1]$):

$$x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}} \cdot (x'_{\max} - x'_{\min}) + x'_{\min}$$

Example: `123, -456, 23, -42` with target $[-1, 1]$ → `1, -1, 0.654, 0.430`

> Min-max is sensitive to outliers because the extreme values define the range. Log or robust scaling (using median and IQR) may be preferred when outliers are present.

### Feature Engineering

Feature extraction creates new variables derived from existing ones, often capturing domain knowledge:

- *Example:* From raw `customer_spending`, compute `spending_delta = spending_t - spending_{t-1}` to capture trends.

### Data Discretisation

Convert continuous variables into categorical bins (useful for some models or interpretability):

- **Equal-width binning:** Divide the range into $k$ equal-sized intervals.
  - E.g., weekly spending → Low [0–10), Medium Low [10–20), Medium [20–30), Medium High [30–40), High [40+)
- **Equal-frequency binning:** Each bin contains the same number of observations.
- Advanced: entropy-based (supervised) or k-means (unsupervised) discretisation.

---

## Data Reduction

For large datasets, reducing the data size can improve both computation speed and, in some cases, model accuracy (by removing noise).

### Sampling

Extract a statistically representative subset:

- **Simple random sampling:** Each record has an equal probability of selection.
- **Stratified sampling:** Divide the population into subgroups (strata) and sample proportionally from each.  
  *Example:* 60% male / 40% female population → select 60 males and 40 females for a sample of 100.

### Feature Selection

Reduce the number of input variables by removing irrelevant or redundant features:

- **Best subset selection:** Evaluate every possible combination — exponential in the number of features.
- **Stepwise selection:** Iteratively add or remove variables based on a criterion (AIC, BIC, p-value).

### Dimensionality Reduction by Projection

Instead of selecting a subset of the original features, **project** onto a lower-dimensional space:

$$[x_1, x_2, \ldots, x_n] \rightarrow [pc_1, pc_2, \ldots, pc_m], \quad m < n$$

Each principal component $pc_j$ is a linear combination of the original features.

**Principal Component Analysis (PCA)** finds the directions of maximum variance in the data and projects observations onto those axes. PCA can reduce, say, 10 correlated variables to 7 uncorrelated components while retaining most of the variance.

---

## Resampling Methods

### The Problem

We want to estimate how well a model will perform on *future, unseen* data. If we evaluate on the same data used to train, we get an over-optimistic estimate (the model has memorised the training data).

### Solution: Resampling

Draw multiple subsets from the training data to simulate the train/test split repeatedly, then average the results.

### Validation Set Approach

Randomly split data into training (~70%) and validation (~30%) sets:

```
500 observations → 350 training | 150 validation
```

- **Advantage:** Simple, easy to implement.
- **Disadvantage:** Estimate is highly variable (depends on which observations land in which split). Models trained on fewer observations tend to perform worse.

### Leave-One-Out Cross-Validation (LOOCV)

Use 1 observation as the validation set and train on the remaining $n-1$. Repeat $n$ times and average the errors.

$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n \text{MSE}_i$$

- **Advantage:** Minimal bias — almost all data is used for training each iteration.
- **Disadvantage:** Computationally expensive (must fit the model $n$ times). LOOCV is a special case of k-fold with $k = n$.

### K-Fold Cross-Validation

Randomly divide the observations into $k$ roughly equal groups (folds):
- Use fold 1 as validation, train on folds 2–k. Repeat $k$ times.
- Final estimate = average of $k$ error rates.

**Typical choice:** $k = 10$ (10-fold CV).

$$\text{CV}_{(k)} = \frac{1}{k}\sum_{i=1}^k \text{MSE}_i$$

- **Advantage:** Good bias-variance tradeoff; computationally cheaper than LOOCV.
- Used extensively in practice for model selection and hyperparameter tuning.

| Method | Bias | Variance | Compute Cost |
|--------|------|----------|-------------|
| Validation Set | High | High | Low |
| LOOCV | Low | Low | Very High |
| k-Fold (k=10) | Moderate | Moderate | Moderate |
