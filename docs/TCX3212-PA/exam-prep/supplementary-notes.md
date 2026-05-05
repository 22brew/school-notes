---
title: "Supplementary Exam Notes"
sidebar_label: "Supplementary Exam Notes"
sidebar_position: 2
---

*Additional concepts from lecture notes that could appear in future exam questions*

---

## Part 1: Python & Pandas (L2, L3)

### 1.1 Key pandas Selection Methods

| Method | What it does | Returns |
|---|---|---|
| `df['col']` | Select single column by label | Series |
| `df[['col']]` | Select single column with double brackets | DataFrame |
| `df[['col1','col2']]` | Select multiple columns | DataFrame |
| `df.loc[row, col]` | Select by **label** | scalar / Series / DataFrame |
| `df.iloc[row, col]` | Select by **integer position** | scalar / Series / DataFrame |
| `df.iloc[:, 0]` | First column by position | Series |
| `df.iloc[0, :]` | First row by position | Series |
| `df.loc[:, 'A':'C']` | All rows, columns A through C (inclusive) | DataFrame |

**Common trap:** `df.loc[[0], 'A':'D']` returns a **DataFrame** (1×4), not a Series, because the row index is passed as a list `[0]`.

### 1.2 GroupBy Mechanics — Full Summary

```python
# Returns DataFrameGroupBy object
g = df.groupby('col')

# Aggregation — Series (single bracket)
df.groupby('day')['tip'].sum()        # Series, indexed by 'day'

# Aggregation — DataFrame (double bracket)
df.groupby('day')[['tip']].sum()      # DataFrame, 'day' as index

# Multiple aggregations
df.groupby('day')['tip'].agg(['mean', 'sum', 'count'])

# Custom agg function
df.groupby('day')['tip'].agg(lambda x: x.quantile(0.75) - x.quantile(0.25))

# Iteration
for key, group_df in df.groupby('day'):
    print(key)        # string key (e.g. 'Fri')
    print(group_df)   # full DataFrame subset
```

**Key facts:**
- `groupby()` alone → returns GroupBy **object** (not Series, not DataFrame)
- `skipna=True` is the **default** in `.mean()`, `.sum()` etc. — NaN values are skipped automatically
- Both `df.groupby('day')['tip'].sum()` and `df['tip'].groupby(df['day']).sum()` give **identical results**

### 1.3 Important pandas Operations

```python
# Missing values
df.isnull()           # boolean mask
df.dropna()           # drop rows with ANY NaN
df.fillna(value)      # fill NaN with value
df.fillna(df.mean())  # fill with column mean

# Sorting
df.sort_values('col', ascending=False)

# Resetting index
df.reset_index(drop=True)   # drop=True: don't add old index as column
df.reset_index()             # adds old index as a new column

# Type conversion
df['col'].astype(str)
df['col'].astype(float)

# pd.Series from Index
pd.Series(df.columns[1:]).astype(str).reset_index(drop=True)
```

### 1.4 matplotlib Plotting Patterns

```python
import matplotlib.pyplot as plt

# Basic subplots setup
fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot (multi-group)
ax.boxplot([group1_data, group2_data], tick_labels=['Group 1', 'Group 2'])

# Essential labels
ax.set_title('Title')
ax.set_xlabel('X-axis label')
ax.set_ylabel('Y-axis label')

plt.tight_layout()
plt.show()

# Barplot
plt.bar(x_values, y_values)

# Line plot
plt.plot(x, y, label='Series Name')
plt.legend()
```

**Boxplot anatomy:**
- Middle line = Median (Q2)
- Box edges = Q1 (25th) and Q3 (75th)
- Whiskers extend to Q1 − 1.5×IQR and Q3 + 1.5×IQR
- Points beyond whiskers = Outliers

---

## Part 2: Data Preparation (L4)

### 2.1 CRISP-DM Data Preparation Pipeline

```
Real-world Data
    → Data Consolidation  (collect, select, integrate)
    → Data Cleaning       (missing values, noise, inconsistencies)
    → Data Transformation (normalize, discretize, construct features)
    → Data Reduction      (reduce variables/cases, balance skewed data)
    → Well-formed Data
```

### 2.2 Handling Missing Values — 4 Strategies

1. **Ignore/Delete** — drop rows/columns with missing values (risks losing data)
2. **Global constant** — replace with "Unknown" or 0 (may introduce bias)
3. **Central tendency** — replace with mean (numerical) or mode (categorical)
4. **Substitution/Imputation** — model-based, hot deck, nearby observations (time series)

```python
df['col'].fillna(df['col'].mean())       # Mean imputation
df['col'].fillna(df['col'].median())     # Median imputation (better for skewed)
df['col'].fillna(method='ffill')         # Forward fill (time series)
df['col'].fillna(method='bfill')         # Backward fill (time series)
```

### 2.3 Data Normalization Formulas

**Min-Max Normalization** — scales to range $[x'_{min}, x'_{max}]$:

$$x'_i = \frac{x_i - x_{min}}{x_{max} - x_{min}} \times (x'_{max} - x'_{min}) + x'_{min}$$

Typically maps to [0, 1] or [−1, 1].

**Decimal Scaling** — shifts decimal by h places:

$$x'_i = \frac{x_i}{10^h}$$

h chosen so all values fall in [−1, 1]. Example: 123, −456 → 0.123, −0.456 (h=3)

**Log Scaling:**

$$x'_i = \log(x_i)$$

Used for right-skewed distributions or to stabilize variance (e.g., income, stock prices).

### 2.4 Outlier Detection

- **Boxplot rule:** Outlier if $x < Q1 - 1.5 \times IQR$ or $x > Q3 + 1.5 \times IQR$
- **Z-score rule:** |z| > 3 (more than 3 standard deviations from mean)
- **Visual inspection:** scatter plots, histograms

---

## Part 3: Linear Regression (L5, L6)

### 3.1 OLS Assumptions (MUST KNOW)

For valid inference (p-values, CIs), the model $Y = \beta_0 + \beta_1 X + u$ requires:

1. **Linearity** — relationship between Y and X is linear in parameters
2. **Normality** — error terms u are normally distributed
3. **Homoscedasticity** — Var(u | X) = σ² (constant variance; errors don't fan out)
4. **Independence** — errors are independent of each other (no autocorrelation)
5. **No perfect multicollinearity** — predictors are not perfect linear combinations of each other

**Time series additionally requires (L10):**
- TS3: Cov(ε_t, ε_s) = 0 for t ≠ s (uncorrelated errors across time)
- TS4: Process is stationary and weakly dependent

### 3.2 OLS Output: Reading Every Number

```
                OLS Regression Results
Dep. Variable:    Y          R-squared:          0.897
Model:            OLS        Adj. R-squared:     0.896
Method:   Least Squares      F-statistic:        570.3
                             Prob (F-statistic):  1.58e-96

              coef  std err        t    P>|t|  [0.025  0.975]
Intercept    2.939    0.312    9.422    0.000   2.324   3.554
TV           0.046    0.001   32.809    0.000   0.043   0.049
```

| Column | Meaning |
|---|---|
| coef | Estimated coefficient β̂ |
| std err | Standard error of β̂ |
| t | t-statistic = coef / std err |
| P>\|t\| | p-value for H₀: β = 0 (two-tailed) |
| [0.025 | Lower bound of 95% CI |
| 0.975] | Upper bound of 95% CI |

### 3.3 Coefficient Interpretation Template

For a continuous predictor X:
> "A one-unit increase in X is associated with a β̂ unit increase/decrease in Y, **on average**, holding all other variables constant."

**Scale matters:** If income is in $1,000s and coefficient = 1.015, then per $1,000 increase in income → +$1,015 in Y (if Y is also in $1,000s).

### 3.4 R² vs Adjusted R² vs F-test — Comparison Table

| Metric | What it measures | Affected by adding variables? |
|---|---|---|
| R² | % of variance in Y explained by the model | Always increases (even irrelevant vars) |
| Adjusted R² | R² penalized for number of predictors | Decreases if added variable is irrelevant |
| F-test p-value | Overall significance: H₀: all β_j = 0 | Tests as a group |
| t-test p-value (each row) | Individual significance of each β_j | Tests one variable at a time |

**Critical distinctions:**
- A significant F-test does NOT mean every variable is significant
- A low R² does NOT mean no variable is significant
- Scaling Y (e.g., minutes → hours) does NOT change R² (scale-invariant)
- AIC/BIC penalize model complexity, useful for model selection

### 3.5 Confidence Intervals — Full Logic

$$\hat{\beta}_j \pm t_{n-p-1, \alpha/2} \times SE(\hat{\beta}_j)$$

| CI contains zero? | Conclusion |
|---|---|
| **No** | Reject H₀ — variable IS significant at that level |
| **Yes** | Fail to reject H₀ — variable is NOT significant at that level |

**Correct CI interpretation:** "We are 95% confident that the true effect of X on Y lies between [lower, upper], holding other variables constant."

**Incorrect:** "The effect is guaranteed/exactly equal to the point estimate."

### 3.6 Collinearity (Multicollinearity)

- Occurs when 2+ predictors are highly correlated with each other
- Effect: inflated standard errors → coefficients become unstable and hard to interpret
- Individual t-tests may show non-significance even if F-test is significant
- **Detection:** Variance Inflation Factor (VIF); correlation matrix between predictors
- **Fix:** Remove one of the correlated variables; use PCA; ridge regression

### 3.7 Common Problems with Linear Regression

| Problem | What to look for | Fix |
|---|---|---|
| Non-linearity | Residual vs fitted plot: curved pattern | Add polynomial terms; log transform |
| Heteroscedasticity | Residuals fan out with fitted values | Log/sqrt transform Y; weighted OLS |
| Autocorrelated errors | Durbin-Watson ≠ 2; time series data | Use time series model; GLS |
| Outliers | Large residuals | Investigate; robust regression |
| High leverage | Extreme X values pulling the fit | Investigate; remove if error |
| Collinearity | High VIF; correlated predictors | Drop variable; PCA |

---

## Part 4: Logistic Regression (L7)

### 4.1 The Full Chain: Probability → Odds → Log-odds

$$p = \Pr(Y=1|X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

$$\text{Odds} = \frac{p}{1-p} = e^{\beta_0 + \beta_1 X}$$

$$\text{Log-odds (logit)} = \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X$$

**The logit is linear in X, but probability is NOT.**

### 4.2 Interpreting Coefficients — All Three Levels

| What you report | How to compute | Template |
|---|---|---|
| Log-odds change | β̂ directly | "A one-unit increase in X increases the log-odds by β̂" |
| Odds ratio | exp(β̂) | "Odds are multiplied by exp(β̂)" |
| % change in odds | \|exp(β̂) − 1\| × 100% | "Odds increase/decrease by X%" |

**Signs:**
- β̂ > 0 → exp(β̂) > 1 → odds increase
- β̂ < 0 → exp(β̂) < 1 → odds decrease

**Do NOT say:** "probability increases by β̂ percentage points" — this is WRONG because the marginal effect on probability is non-constant (depends on current value of X).

### 4.3 Multi-unit Changes

For a k-unit increase in X:
- Log-odds change by $k \times \hat{\beta}$
- Odds are multiplied by $\exp(k \times \hat{\beta})$

Example: β̂ = 0.0302 for MonthlyCharges. A $10 increase:
- Log-odds change: 10 × 0.0302 = 0.302
- Odds multiplied by: exp(0.302) ≈ 1.353 → 35.3% increase

### 4.4 Interaction Terms in Logistic Regression

Model: $\text{logit}(p) = \beta_0 + \beta_1 \cdot \text{FamHist} + \beta_2 \cdot \text{Age} + \beta_3 \cdot \text{Worry} + \beta_4 \cdot (\text{Worry} \times \text{FamHist})$

| Condition | Effect of Worry |
|---|---|
| FamHist = 0 | $\exp(\beta_3)$ |
| FamHist = 1 | $\exp(\beta_3 + \beta_4)$ |

The interaction coefficient β₄ represents **how much the effect of Worry changes** when moving from no family history to having family history.

### 4.5 Confusion Matrix — Full Reference

```
                  Actual: YES (1)    Actual: NO (0)
Predicted: YES        TP                 FP
Predicted: NO         FN                 TN
```

| Metric | Formula | Plain English |
|---|---|---|
| Accuracy | (TP+TN)/(TP+FP+FN+TN) | % of all predictions that are correct |
| Sensitivity / Recall / TPR | TP/(TP+FN) | Of all actual positives, how many did we catch? |
| Specificity / TNR | TN/(TN+FP) | Of all actual negatives, how many did we correctly exclude? |
| Precision / PPV | TP/(TP+FP) | Of all predicted positives, how many are truly positive? |
| F1-Score | 2×Prec×Sens/(Prec+Sens) | Harmonic mean of precision and sensitivity |
| False Positive Rate | FP/(FP+TN) = 1 − Specificity | Type I error rate |
| False Negative Rate | FN/(FN+TP) = 1 − Sensitivity | Type II error rate |

**When each metric matters:**
- **High Sensitivity needed:** Medical screening (don't miss sick people), fraud detection
- **High Specificity needed:** Spam filters (don't incorrectly block legitimate mail)
- **High Accuracy:** Balanced classes where both types of error are equally costly
- **High Precision:** When false alarms are costly (e.g., targeted marketing)

### 4.6 Pseudo R² in Logistic Regression

- Logistic regression does NOT have a true R²
- **McFadden's Pseudo R²** — 0.003–0.05 is considered poor; 0.2–0.4 is very good
- Much lower values are typical compared to OLS R²
- **LLR p-value:** Tests H₀ that all coefficients = 0 (analogous to F-test)

---

## Part 5: Time Series (L8, L9, L10)

### 5.1 Time Series Components

| Component | Description | Example |
|---|---|---|
| Trend ($m_t$) | Long-run upward/downward movement | Rising sales over years |
| Seasonality ($s_t$) | Regular periodic fluctuation | Monthly retail, L seasons per year |
| Irregular ($Y_t$) | Random noise (stationary process) | Random shocks |

Classical decomposition:
- $X_t = m_t + s_t + Y_t$ (Additive)
- $X_t = m_t \times s_t \times Y_t$ (Multiplicative — use when seasonal variation grows with level)

**Use multiplicative when:** seasonal swings grow proportionally with the trend level (e.g., airline passengers).

### 5.2 Stationarity — Essential Concept

A process is **stationary** if:
- Mean is constant over time
- Variance is constant over time
- Autocovariance depends only on lag h, not on time t

**Ways stationarity can FAIL:**
- Trend (mean changes over time)
- Seasonality (periodic mean changes)
- Sudden structural break (regime change, crisis)
- Non-constant variance (heteroscedasticity in time)

**Fixes:**
- Trend: First differencing: $y'_t = y_t - y_{t-1}$
- Seasonality: Seasonal differencing: $y'_t = y_t - y_{t-m}$ (m = seasons per year)
- Increasing variance: Log transformation

### 5.3 Differencing

**First-order differencing** (removes linear trend):

$$\nabla y_t = y_t - y_{t-1}$$

**Seasonal differencing** (removes seasonality, lag = m):

$$\nabla_m y_t = y_t - y_{t-m}$$

where m = 12 for monthly data, m = 4 for quarterly data.

### 5.4 Moving Averages (SMA)

$$SMA_t = \frac{y_t + y_{t-1} + \cdots + y_{t-w+1}}{w}$$

| Window width w | Effect |
|---|---|
| Large w | Smooth, slow to react to changes |
| Small w | Noisy, fast to react to changes |

**SMA is a smoothing tool, not a forecasting model** (it lags behind turning points).

### 5.5 Exponential Smoothing — All Three Methods

#### Simple Exponential Smoothing (SES) — No trend, no seasonality

$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$$

- α near 1 → more weight on recent observations (reacts quickly)
- α near 0 → more weight on past observations (smoother)
- Forecast h steps ahead: $\hat{y}_{t+h} = \hat{y}_{t+1}$ (flat line)

#### Double Exponential Smoothing (Holt's) — Trend, no seasonality

$$a_t = \alpha y_t + (1-\alpha)(a_{t-1} + b_{t-1})$$

$$b_t = \beta(a_t - a_{t-1}) + (1-\beta)b_{t-1}$$

$$\hat{y}_{t+h} = a_t + h \cdot b_t$$

- $a_t$ = **level** (current estimated value)
- $b_t$ = **trend** (current estimated slope)
- Higher β → more sensitive to recent trend changes
- Forecast is a **straight line** projecting from $a_t$ with slope $b_t$

#### Holt-Winters (Triple) — Trend AND seasonality

$$a_t = \alpha(y_t - c_{t-L}) + (1-\alpha)(a_{t-1} + b_{t-1})$$

$$b_t = \beta(a_t - a_{t-1}) + (1-\beta)b_{t-1}$$

$$c_t = \gamma(y_t - a_t) + (1-\gamma)c_{t-L}$$

$$\hat{y}_{t+h} = a_t + h \cdot b_t + c_{t+h-L}$$

- L = number of seasons (12 for monthly, 4 for quarterly)
- $c_t$ = seasonal index
- Three parameters: α (level), β (trend), γ (seasonality)

**Choosing the right model:**

| Series has... | Use |
|---|---|
| No trend, no seasonality | Simple Exponential Smoothing |
| Trend, no seasonality | Double Exponential Smoothing (Holt's) |
| Trend AND seasonality | Holt-Winters (Triple) |

### 5.6 Forecast Performance Metrics

| Metric | Formula | Notes |
|---|---|---|
| MAE | (1/n) Σ|e| | Average absolute error; same units as y |
| MSE | (1/n) Σe² | Penalizes large errors more |
| RMSE | √(MSE) | Same units as y; comparable to MAE |
| MAPE | (1/n) Σ(|e|/y) × 100% | % error; scale-independent |

**Forecast error:** $e_t = y_t - \hat{y}_t$ (actual − predicted)

**Training vs Test data:**
- Training (in-sample): data used to fit the model
- Test (out-of-sample): unseen future data used to evaluate forecast accuracy
- **Always evaluate on test set** to avoid overfitting

### 5.7 ACF / Correlogram — Interpretation Guide

$$\rho_X(h) = \text{corr}(X_t, X_{t+h}) = \frac{\gamma_X(h)}{\gamma_X(0)}$$

| What ACF looks like | What it means |
|---|---|
| All lags ≈ 0 (within dashed lines) | White noise — no autocorrelation |
| Slow decay (gradually decreasing) | Strong trend; non-stationary |
| Alternating positive/negative | Oscillating series |
| Spike at lag h, then drops | MA(h) pattern |
| Gradual decay | AR pattern |
| Repeating spikes at multiples of m | Seasonality with period m |

**Dashed lines** = ±1.96/√n (95% significance bounds for white noise)  
Values outside = statistically significant autocorrelation at that lag.

### 5.8 Regression-Based Time Series Models (L10)

#### Static Model

$$y_t = \beta_0 + \beta_1 x_{1t} + \beta_2 x_{2t} + \cdots + \beta_k x_{kt} + \epsilon_t$$

#### Time Series with Trend

$$y_t = \beta_0 + \beta_1 t + \epsilon_t \quad \text{(linear trend)}$$

$$y_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \epsilon_t \quad \text{(quadratic trend)}$$

#### Finite Distributed Lag (FDL) Model

$$y_t = \beta_0 + \delta_0 x_t + \delta_1 x_{t-1} + \delta_2 x_{t-2} + \epsilon_t$$

- δ₀ = **impact multiplier**: immediate effect of x on y
- δ₁ = **1-period dynamic multiplier**: effect 1 period later
- δ₂ = **2-period dynamic multiplier**: effect 2 periods later
- δ₀ + δ₁ + δ₂ = **long-run multiplier** (cumulative effect)

#### Autoregressive Model AR(1)

$$y_t = \beta_0 + \beta_1 y_{t-1} + u_t$$

- If β₁ is significant → past values help predict future values
- If β₁ is NOT significant → consistent with white noise / random walk

---

## Part 6: Regression Advanced Topics (L6)

### 6.1 Polynomial Regression

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + u$$

**Marginal effect** (change in y per unit change in x):

$$\frac{\partial y}{\partial x} = \beta_1 + 2\beta_2 x$$

This is NOT constant — it depends on the current value of x.

**Signs of β₂:**
- β₂ > 0 → U-shaped (concave up); effect increases with x
- β₂ < 0 → Inverted U-shape (concave down); effect decreases with x (diminishing returns)

**Turning point** (where marginal effect = 0): $x^* = -\beta_1 / (2\beta_2)$

### 6.2 Interaction Terms — All Cases

**Continuous × Continuous:**

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2$$

- Effect of x₁ on y: β₁ + β₃·x₂ (varies with level of x₂)
- Effect of x₂ on y: β₂ + β₃·x₁ (varies with level of x₁)

**Continuous × Binary dummy:**

$$y = \beta_0 + \beta_1 x + \beta_2 D + \beta_3 (x \cdot D)$$

- For D = 0 (reference): effect of x = β₁
- For D = 1: effect of x = β₁ + β₃
- β₃ = how much the slope of x changes for D=1 vs D=0

**Always take the partial derivative ∂y/∂x to find the marginal effect.**

### 6.3 Coding Categorical Variables

- k categories → k−1 dummy variables
- Reference (omitted) category = when all dummies = 0
- Intercept captures the mean of the reference group
- Each dummy coefficient = deviation from the reference group mean

```python
import statsmodels.formula.api as smf
smf.ols('y ~ C(gender, Treatment(reference="Female"))', data=df).fit()
```

### 6.4 F-test Variants

**Overall F-test** (standard):
- H₀: β₁ = β₂ = … = β_k = 0 (all slope coefficients are zero)
- H₁: At least one β_j ≠ 0

**Partial F-test** (subset of coefficients):
- Tests whether a subset of variables jointly contribute to the model
- Used when comparing a restricted vs. unrestricted model

---

## Part 7: Tricky MCQ Traps — Common Wrong Answers

### Trap 1: F-test vs Individual t-tests

- A significant F-test (p < 0.05) means **at least one** variable is significant
- It does NOT mean **all** variables are significant
- Each variable needs its own p-value < 0.05 to be individually significant

### Trap 2: Logistic Regression Marginal Effect on Probability

- The coefficient β̂ gives the change in **log-odds** per unit change in X
- The change in **probability** is NOT constant — it depends on the current X value
- "Each unit increase increases probability by β̂%" → **WRONG**
- "Each unit increase multiplies odds by exp(β̂)" → **CORRECT**

### Trap 3: Confidence Interval Contains Zero

- CI contains zero → fail to reject H₀ → NOT significant
- The asymmetry of CI bounds (|lower| > |upper|) does NOT make it significant
- Only the presence or absence of zero matters

### Trap 4: Holt-Winters β parameter

- Higher β → **MORE** sensitive to recent trend changes (not less)
- Higher α → **MORE** weight on recent observations
- $a_t$ is the **level** (intercept of the forecast line)
- $b_t$ is the **trend/slope** (rate of change)

### Trap 5: R² and Scaling

- Multiplying Y by a constant → R² stays the **same** (scale-invariant)
- Adding an irrelevant variable → R² stays same or **increases** (never decreases)
- Adding an irrelevant variable → Adjusted R² will **decrease** (if variable adds nothing)

### Trap 6: Dummy Variable Interpretation

- Intercept (β₀) = mean of the **reference group**
- Dummy coefficient = **difference** from reference group, NOT the absolute mean of that group
- b₁ = afternoon − morning (NOT the average in afternoon)

### Trap 7: GroupBy Object Type

- `df.groupby('col')` → returns **GroupBy object**, NOT a Series or DataFrame
- `df.groupby('col')['val'].sum()` → Series
- `df.groupby('col')[['val']].sum()` → DataFrame

### Trap 8: Log-odds vs Odds vs Probability

| Layer | Expression | Interpretation |
|---|---|---|
| Log-odds | β₀ + β₁X | Linear; changes by β₁ per unit X |
| Odds | exp(β₀ + β₁X) | Multiplicative; × exp(β₁) per unit X |
| Probability | sigmoid(linear) | Non-linear; marginal effect varies |

### Trap 9: ACF at Lag 0

- Always exactly 1.0 — mathematical definition, not an estimate
- NOT related to stationarity, SNR, or Ljung-Box test

### Trap 10: Interaction Terms in Logistic Regression

For $\text{logit}(p) = \beta_0 + \beta_1 \cdot \text{FamHist} + \beta_2 \cdot \text{Worry} + \beta_3 \cdot (\text{Worry} \times \text{FamHist})$:
- **No FamHist (=0):** effect of Worry = exp(β₂)
- **With FamHist (=1):** effect of Worry = exp(β₂ + β₃) — must add BOTH terms

---

## Part 8: Model Selection Criteria

### AIC and BIC

$$AIC = -2\ln(L) + 2k$$

$$BIC = -2\ln(L) + k\ln(n)$$

where L = likelihood, k = number of parameters, n = sample size.

- **Lower is better** for both AIC and BIC
- BIC penalizes complexity more than AIC (because of ln(n) factor)
- Use AIC when prediction is the goal; BIC when parsimony matters

### Model Comparison Summary

| Metric | Goal | More predictors → |
|---|---|---|
| R² | Goodness of fit | Always increases |
| Adjusted R² | Penalized fit | Increases only if variable is useful |
| AIC | Prediction | Usually decreases with useful vars |
| BIC | Parsimony | More strictly penalizes complexity |
| F-test | Overall significance | Tests H₀: all β = 0 |

---

## Part 9: Python Code Patterns for Essays

### 9.1 OLS Regression (statsmodels)

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Simple
model = smf.ols('y ~ x', data=df).fit()
print(model.summary())

# Multiple
model = smf.ols('y ~ x1 + x2 + x3', data=df).fit()

# With dummy (categorical)
model = smf.ols('y ~ C(group)', data=df).fit()

# With interaction
model = smf.ols('y ~ x1 * x2', data=df).fit()  # includes x1, x2, x1:x2

# Confidence intervals
model.conf_int()  # returns DataFrame of [0.025, 0.975] CIs
```

### 9.2 Logistic Regression (statsmodels)

```python
model = smf.logit('y ~ x1 + x2', data=df).fit()
print(model.summary())

# Predicted probabilities
probs = model.predict(df)

# Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = (probs >= 0.5).astype(int)
cm = confusion_matrix(df['y'], y_pred)
```

### 9.3 Descriptive Statistics & IQR

```python
# Summary stats
df['col'].describe()

# IQR
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1

# Custom agg function
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

df.groupby('group')['value'].agg(['mean', 'median', 'std', iqr])
```

### 9.4 Time Series (statsmodels / pandas)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple Exponential Smoothing
model = ExponentialSmoothing(y, trend=None, seasonal=None)
fit = model.fit(smoothing_level=0.3, optimized=False)

# Double (Holt's)
model = ExponentialSmoothing(y, trend='add', seasonal=None)
fit = model.fit(smoothing_level=0.3, smoothing_trend=0.2, optimized=False)

# Triple (Holt-Winters)
model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit(smoothing_level=0.3, smoothing_trend=0.2, smoothing_seasonal=0.1, optimized=False)

# Forecast
forecast = fit.forecast(steps=12)

# ACF plot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y, lags=40)

# Differencing
y_diff = y.diff(1)    # first difference
y_seas = y.diff(12)   # seasonal difference (monthly)
```

---

## Part 10: Rapid Calculation Reference

### Logistic Regression Quick Calculations

| Given | Compute | Formula |
|---|---|---|
| β̂ | Odds ratio | exp(β̂) |
| β̂ | % odds change | (exp(β̂) − 1) × 100% |
| k-unit change in X, β̂ | Odds multiplier | exp(k × β̂) |
| β₀, β₁, X value | Probability | exp(η)/(1+exp(η)) where η = β₀+β₁X |
| No family hist coef β₃, interaction coef β₄ | Effect for FamHist=1 | exp(β₃ + β₄) |

### Holt-Winters Quick Calculations

$$b_t = \beta(a_t - a_{t-1}) + (1-\beta)b_{t-1}$$

$$\hat{y}_{t+h} = a_t + h \cdot b_t$$

**Worked example steps:**
1. Compute $(a_t - a_{t-1})$ — "recent trend"
2. Multiply by β
3. Multiply $b_{t-1}$ by $(1 - \beta)$ — "old trend weighted"
4. Add both parts → new $b_t$

### Marginal Effect of Quadratic Model

$$\text{return}_t = \beta_0 + \beta_1 \text{ret}_{t-1} + \beta_2 \text{ret}_{t-1}^2$$

$$\frac{\partial y}{\partial x} = \beta_1 + 2\beta_2 \cdot x \quad \text{(evaluate at given x)}$$

### Confusion Matrix Calculations

```
          Actual 1   Actual 0
Pred 1      TP         FP
Pred 0      FN         TN
```

| Metric | Formula |
|---|---|
| Accuracy | (TP+TN)/(TP+FP+FN+TN) |
| Sensitivity | TP/(TP+FN) |
| Specificity | TN/(TN+FP) |
| Precision | TP/(TP+FP) |

---

*Sources: TCX3212 Lectures 1–10 | Key supplementary topics for exam preparation*
