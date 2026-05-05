---
title: "Finals Study Reference"
sidebar_label: "Finals Study Reference"
sidebar_position: 1
---

*Covers L2–L10: Python Programming, Regression, Logistic Regression, Time Series*

---

## Q1 — Pandas GroupBy (L2)

**Answer: C (i, ii, and v)**

### Key Concepts

- `tips.groupby('day')['tip'].sum()` → groups first, then selects column → returns a **Series** indexed by 'day'
- `tips.groupby('day')[['tip']].sum()` → double bracket `[[...]]` → returns a **DataFrame** with 'day' as index
- `tips['tip'].groupby(tips['day']).sum()` → selects Series first, then groups → **same result** as single-bracket groupby
- `tips.groupby('day')` → returns a **DataFrameGroupBy object**, NOT a Series (so iv is FALSE)
- Iterating: `for day, group in tips.groupby('day')` → `day` = group key (string), `group` = DataFrame of that subset
- `.mean()` by default uses `skipna=True`, so it will NOT raise an error on NaN (iii is FALSE)

---

## Q2 — DataFrame Index/Column Extraction (L2 & L3)

**Answer: E (iii only)**

### Key Concepts

- `df.columns[1:]` → excludes 'Data Series', returns an Index object (not a Series)
- Wrap with `pd.Series(...)` to convert to Series, then `.astype(str).reset_index(drop=True)`
- `df.iloc[:, 0]` → selects first column **by position** → correct for getting row headers
- `df.loc[:, 0]` → selects by **label** 0 (only works if the column is named 0)
- `.reset_index()` without `drop=True` → adds old index as a column (makes a DataFrame, not a clean Series)

**Correct code:**
```python
years = pd.Series(df.columns[1:]).astype(str).reset_index(drop=True)
headers = df.iloc[:, 0].astype(str).reset_index(drop=True)
```

---

## Q3 — OLS Output: Statistical Significance (L5 & L6)

**Answer: A (ii and v)**

### Key Concepts: Reading OLS Output

| Variable | p-value | Significant at... |
|---|---|---|
| totwrk | 0.000 | 1%, 5%, 10% ✓ |
| income | 0.445 | Not significant at 10% |
| educ | 0.114 | Not significant at 10% |
| age | 0.105 | Not significant at 5% or 10% (borderline) |

- **F-test p-value** (1.48e-17) → the model as a whole is significant (at least one β ≠ 0). This does NOT mean all individual variables are significant (so i is FALSE)
- **Individual t-tests** use each variable's own p-value, not the F-test
- ii ✓: Only totwrk is significant at 1%
- v ✓: totwrk significant at 1%; income/educ/age not significant at 5%; F-test shows model overall is significant

---

## Q4 — Confidence Intervals (L6)

**Answer: E (i, iii, and v)**

### Key Concepts: Confidence Intervals

- A 95% CI that **does not contain zero** → reject H₀ at 5% significance → variable IS significant
- A 95% CI that **contains zero** → fail to reject H₀ → variable is NOT significant at 5%

| Variable | 95% CI | Zero Included? | Significant? |
|---|---|---|---|
| totwrk | [-0.180, -0.114] | No | Yes ✓ |
| educ | [-21.840, 2.355] | Yes | No (ii is FALSE — CI containing zero means NOT significant) |
| age | [-0.499, 5.252] | Yes | No ✓ (iii correct) |
| income | [-0.005, 0.002] | Yes | No |

- i ✓: totwrk CI excludes zero → reject H₀
- iii ✓: age CI contains zero → fail to reject H₀ → not significant at 5%
- v ✓: Correct interpretation of CI → "95% confident the decrease is between 0.114 and 0.180 minutes"
- iv is FALSE: A CI gives a range of plausible values, not a guaranteed exact effect

---

## Q5 — Goodness of Fit: R² and Adjusted R² (L5)

**Answer: A (i and ii)**

### Key Concepts

- **R²** always stays the same or increases when a variable is added (even irrelevant ones) → i ✓
- **Adjusted R²** penalizes for adding irrelevant variables → decreases if the new variable adds no explanatory power → i ✓
- **R² = 0.114** → 11.4% of variance in sleep is explained by the model → ii ✓
- Low R² does NOT mean the model is useless; a variable can be statistically significant with low R² (iii is FALSE)
- **F-test H₀**: ALL slope coefficients = 0 (not "at least one equals zero" → iv is FALSE)
- Scaling the dependent variable does NOT change R² (v is FALSE) → R² is scale-invariant

---

## Q6 — Logistic Regression: Odds Interpretation (L7)

**Answer: C (i and iii)**

### Key Concepts: Interpreting Logistic Regression Coefficients

- A coefficient β means: a one-unit increase in X multiplies the odds by **exp(β)**
- **% change in odds** = |exp(β) − 1| × 100%

| Variable | Coefficient | exp(coef) | Interpretation |
|---|---|---|---|
| Tenure | -0.0671 | exp(-0.0671) ≈ 0.935 | Each month reduces odds of churn by ~6.5% |
| MonthlyCharges | 0.0302 | exp(0.0302×10) = exp(0.302) ≈ 1.35 | $10 increase → odds × 1.35 (35% increase) |

- i ✓: $10 increase → exp(0.302) ≈ 1.35 → 35% increase in odds
- iii ✓: Each month tenure → odds × 0.935 → 6.5% decrease
- iv is FALSE: The coefficient gives log-odds change, not a fixed probability change
- v is FALSE: Logistic regression has a non-linear probability function, so marginal effect on probability is NOT constant

---

## Q7 — Dummy Variables in Regression (L6)

**Answer: E (i, iii, and iv are true)**

### Key Concepts: Dummy Variable Regression

Model: `NUMBER_CARS = b0 + b1*AFTERNOON + b2*EVENING`

| Setting | Equation | Interpretation |
|---|---|---|
| MORNING (both = 0) | = b0 | b0 = average cars in morning |
| AFTERNOON (= 1, EVENING = 0) | = b0 + b1 | b1 = **difference** in average cars, afternoon vs. morning |
| EVENING (AFTERNOON = 0, EVENING = 1) | = b0 + b2 | b2 = **difference** in average cars, evening vs. morning |

- i ✓: b0 = average cars in morning (reference group intercept)
- ii ✗: b1 is the **difference** between afternoon and morning, NOT the average in afternoon
- iii ✓: b2 = difference (evening − morning)
- iv ✓: b0 is a count of cars → must be positive (number of cars cannot be negative)
- v ✓: b1 could be negative if fewer cars in afternoon

---

## Q8 — Holt-Winters / Double Exponential Smoothing (L9)

**Answer: D (i only)**

### Key Concepts: Trend Equation

$$b_t = \beta(a_t - a_{t-1}) + (1-\beta)b_{t-1}$$

**Calculation:**
- $a_t = 215$, $a_{t-1} = 200$, $b_{t-1} = 10$, $\beta = 0.3$
- $b_t = 0.3 \times (215 - 200) + (1 - 0.3) \times 10$
- $b_t = 0.3 \times 15 + 0.7 \times 10 = 4.5 + 7.0 = \mathbf{11.5}$ ✓ (i is correct)

**Why others are wrong:**
- ii: If β = 0.7: $b_t = 0.7 \times 15 + 0.3 \times 10 = 10.5 + 3.0 = \mathbf{13.5}$, not 14.0. Also, higher β = MORE sensitive to recent changes, not less.
- iii: In $\hat{y}_{t+h} = a_t + h \cdot b_t$, **$a_t$ is the level (intercept)** and **$b_t$ is the slope/trend** — they are swapped in statement iii.
- iv: If $a_t = 210 = a_{t-1} + b_{t-1} = 200 + 10$, then $b_t = 0.3 \times (210-200) + 0.7 \times 10 = 3 + 7 = 10$ (stays the same, not decreases).

---

## Q9 — ACF at Lag 0 (L8)

**Answer: A (i and iii only)**

### Key Concepts: Autocorrelation Function (ACF)

$$\rho_X(h) = \frac{\gamma_X(h)}{\gamma_X(0)} = \text{corr}(X_t, X_{t+h})$$

At **lag h = 0**:
- We are computing corr($X_t$, $X_t$) = correlation of a variable with itself
- Any variable has perfect correlation with itself = **1.0**
- This is a mathematical certainty, not a statistical estimate

- i ✓: Dimensionless; at lag 0, correlation of $X_t$ with itself = 1 (perfect)
- iii ✓: Any random variable is a perfect linear function of itself → correlation = 1
- ii ✗: The value at lag 0 is definitionally 1 → it has nothing to do with the Ljung-Box test
- iv ✗: Non-stationarity is about mean/variance changing over time, not about ACF at lag 0
- v ✗: ACF is NOT a signal-to-noise ratio

---

## Q10 — Logistic Regression Predicted Probability (L7)

**Answer: E**

### Key Concepts: Logistic Regression Probability Formula

$$p = \frac{e^{\hat{\beta}_0 + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2 + \hat{\beta}_3 X_3}}{1 + e^{\hat{\beta}_0 + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2 + \hat{\beta}_3 X_3}}$$

**Given:** Intercept = −6.38, Prostate marker β = 0.02, Age β = −0.02, Gleason β = 1.07  
**Patient:** Age = 69, Marker = 10, Gleason = 5

$$p = \frac{\exp(-6.38 + 0.02 \times 10 - 0.02 \times 69 + 1.07 \times 5)}{1 + \exp(-6.38 + 0.02 \times 10 - 0.02 \times 69 + 1.07 \times 5)}$$

**Why other options are wrong:**
- A: Missing the intercept and age terms
- B: Returns the raw exponent, not bounded [0,1]
- C: log(exp(...)) just gives back the linear predictor
- D: log(sigmoid) → this is the log-probability, not probability

---

## Q11 — Linear Regression: Dummy Variable Coefficient Interpretation (L6)

**Answer: A (i only)**

### Key Concepts

Reference group = Female (coded 0). gender1 = Male (coded 1).

The coefficient for gender1 = **650.48** means:
> "Males have, on average, 650.48 units higher total plasma levels than females, holding all other variables constant."

- i ✓: Correct interpretation — males vs. reference (females), holding other variables constant
- ii ✗: A coefficient does not "explain units of variation" — that's what R² does
- iii ✗: The direction is reversed — females are the reference, so males are HIGHER
- iv ✗: 6.51% would be a partial R² concept, not coefficient interpretation
- v ✗: The direction should specify male vs. female (not ambiguous "one-unit increase")

---

## Q12 — Logistic Regression with Interaction Term (L7)

**Answer: A (i, iii, and iv only)**

### Key Concepts: Interaction Terms in Logistic Regression

Model: $\text{logit}(p) = \beta_0 + \beta_1(\text{FamilyHistory}) + \beta_2(\text{Age}) + \beta_3(\text{Worry}) + \beta_4(\text{Worry} \times \text{FamilyHistory})$

**Coefficients:**
- Age: −0.04 → exp(−0.04) ≈ 0.961 → each year, odds × 0.961 (≈ 4% decrease, NOT 96% increase)
- Worry: −0.04
- Worry × FamilyHistory: −0.05

**For women with NO family history (FamilyHistory = 0):**
- Effect of worry = β₃ = −0.04 → exp(−0.04) ≈ 0.96 → iii ✓

**For women WITH family history (FamilyHistory = 1):**
- Effect of worry = β₃ + β₄ = −0.04 + (−0.05) = −0.09 → exp(−0.09) ≈ 0.91 → iv ✓

- i ✓: Age coefficient = −0.04 → exp(−0.04) ≈ 0.961, so odds decrease by factor of ~0.96 per year
- ii ✗: Family history coefficient = 0.1465, p = 0.7555 → NOT significant; also exp(0.1465) ≈ 1.158, so ~16% increase, not 25%
- v ✗: exp(−0.04) ≈ 0.96 means odds are MULTIPLIED by 0.96 (decrease), not increase by 96%

---

## Q13 — Python: Side-by-Side Boxplots (L3)

**Topic: matplotlib, groupby, visualization**

### Sample Answer

```python
import matplotlib.pyplot as plt

male_bills = tips[tips['sex'] == 'Male']['total_bill']
female_bills = tips[tips['sex'] == 'Female']['total_bill']

fig, ax = plt.subplots()
ax.boxplot([male_bills, female_bills], tick_labels=['Male', 'Female'])
ax.set_title('Distribution of Total Bill by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Total Bill ($)')
plt.show()
```

**Key requirements:**
- Each gender has its own separate boxplot
- Title, x-axis label (Gender), y-axis label (Total Bill)
- Allows comparison of medians, spread (IQR), and outliers

---

## Q14 — Python: analyze_spending Function (L2 & L3)

**Topic: pandas groupby, aggregation, IQR**

### Sample Answer

```python
import pandas as pd

def analyze_spending(df, group_col, value_col):
    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    
    result = df.groupby(group_col)[value_col].agg(
        Mean='mean',
        Median='median',
        Std='std',
        IQR=iqr
    ).reset_index()
    
    return result
```

**Key points:**
- `groupby(group_col)` groups by the categorical column
- `.agg()` applies multiple aggregation functions simultaneously
- IQR = Q3 − Q1 = `quantile(0.75) − quantile(0.25)`
- Returns a new DataFrame with one row per group

---

## Q15 & Q16 — OLS with Interaction Term: 401k Dataset (L6)

**Model:** $\text{nettfa} = \beta_0 + \beta_1 \cdot \text{income} + \beta_2 \cdot \text{age} + \beta_3 \cdot \text{p401k1} + \beta_4 \cdot (\text{age} \times \text{p401k1})$

### OLS Output Summary

| Variable | Coef | Std Err | t | P>\|t\| | Significant at 5%? |
|---|---|---|---|---|---|
| Intercept | -58.29 | 7.902 | -7.377 | 0.000 | Yes |
| age | 0.801 | 0.184 | 4.360 | 0.000 | **Yes** ✓ |
| income | 1.015 | 0.071 | 14.276 | 0.000 | Yes |
| p401k1 | -10.01 | 16.250 | -0.616 | 0.538 | **No** |
| age:p401k1 | 0.6517 | 0.377 | 1.729 | 0.084 | No |

### Q15(i) — Statistical Significance Test

**Decision rule:** Reject H₀ if p-value < 0.05

- **age**: p = 0.000 < 0.05 → **Statistically significant** at 5%. Reject H₀ (β_age = 0).
- **p401k1**: p = 0.538 > 0.05 → **NOT statistically significant** at 5%. Fail to reject H₀.

### Q15(ii) — Interpretation of Income Coefficient

Income is measured in **thousands of dollars** ($1,000s):
- Coefficient = 1.015 (per $1,000 of income)
- Per $1 of income: 1.015 / 1000 = **$0.001015 increase in nettfa** (also in $1,000s)
- In dollar terms: For every $1 increase in annual income, net total financial assets increase by approximately **$1.02** on average, holding age, 401(k) participation, and the interaction term constant.

---

## Q16 — Interaction Term Interpretation (L6)

**Model:** $\text{nettfa} = \beta_0 + \beta_1 \cdot \text{income} + \beta_2 \cdot \text{age} + \beta_3 \cdot \text{p401k1} + \beta_4 \cdot (\text{age} \times \text{p401k1})$

### (i) Marginal Effect of Age

$$\frac{\partial \text{nettfa}}{\partial \text{age}} = \beta_2 + \beta_4 \cdot \text{p401k1}$$

**For non-401(k) participants (p401k1 = 0):**
- Marginal effect = β₂ = **0.801**
- Each additional year of age is associated with a $0.80K → **$801** increase in net financial assets

**For 401(k) participants (p401k1 = 1):**
- Marginal effect = β₂ + β₄ = 0.801 + 0.6517 = **1.4527**
- Each additional year of age is associated with approximately **$1,453** increase in net financial assets

**Conclusion:** Age has a larger positive effect on net financial assets for those who participate in 401(k) plans.

### (ii) Statistical Significance at 1%

- p-value for age:p401k1 = **0.084**
- At 1% significance level: p = 0.084 > 0.01 → **Fail to reject H₀**
- The interaction term is **NOT statistically significant at the 1% level**

---

## Q17 — Logistic Regression: Eligibility & Confusion Matrix (L7)

**Model:** $\text{Logit}(\text{Eligibility\_Binary}) = \beta_0 + \beta_1 \cdot \text{age} + \beta_2 \cdot \text{gender1}$

### Logit Output

| Variable | Coef | p-value |
|---|---|---|
| Intercept | -0.9010 | 0.000 |
| age | 0.0120 | 0.028 |
| gender1 | -0.0925 | 0.518 |

### (i) Interpretation of gender1

Reference group = Female (0). gender1 = Male (1).
- Coefficient = −0.0925; exp(−0.0925) ≈ 0.912
- **Interpretation:** Being male is associated with a multiplicative factor of approximately 0.912 on the odds of being eligible, compared to females, holding age constant — about an **8.8% decrease** in odds. However, gender1 is NOT statistically significant (p = 0.518 > 0.05).

### (ii) Confusion Matrix Evaluation

| | Reference: 0 | Reference: 1 |
|---|---|---|
| Predicted 0 | TN = 496 | FN = 285 |
| Predicted 1 | FP = 305 | TP = 239 |

Total = 496 + 285 + 305 + 239 = **1325**

| Metric | Calculation | Result |
|---|---|---|
| **Accuracy** | (239 + 496) / 1325 | ≈ **55.5%** |
| **Sensitivity** | 239 / (239 + 285) | ≈ **45.6%** |
| **Specificity** | 496 / (496 + 305) | ≈ **61.9%** |

**Business Context:** A financial firm launching a savings intervention would prioritize high **sensitivity** (catch as many eligible people as possible). The current sensitivity of ~45.6% is too low — the model would miss more than half of its intended audience.

---

## Q18 — Time Series Regression: EMH Analysis (L10)

**Model:** $\text{return}_t = \beta_0 + \beta_1 \cdot \text{return}_{t-1} + u_t$

### OLS Output

| Variable | Coef | Std Err | t | P>\|t\| | 95% CI |
|---|---|---|---|---|---|
| Intercept (const) | 0.1796 | 0.0807 | 2.225 | 0.026 | [0.021, 0.338] |
| return_1 | 0.0589 | 0.0380 | 1.549 | 0.122 | [-0.016, 0.134] |

### (i) Fitted Regression Equation

$$\widehat{\text{return}}_t = 0.18 + 0.06 \cdot \text{return}_{t-1}$$

*(Rounded to 2 d.p. as instructed)*

### (ii) Interpretation of return_1 Coefficient (0.0589)

- For every 1 percentage point increase in last week's market return, this week's return is expected to increase by approximately **0.06 percentage points** on average.
- The positive sign suggests a slight momentum effect — positive returns last week are associated with slightly higher returns this week.
- However, the **magnitude is very small**, and the relationship may not be practically meaningful.

### (iii) Does the Data Support the EMH?

**Test:** H₀: β₁ = 0 (last week's return has no predictive power → supports EMH)
- p-value for return_1 = **0.122 > 0.05**
- **Fail to reject H₀** at the 5% significance level

**Conclusion:** The data **supports the EMH**. There is insufficient statistical evidence that last week's return helps predict this week's return (p = 0.122). The lagged return variable is not statistically significant, consistent with the hypothesis that past price information cannot reliably predict future returns.

---

## Q19 — Time Series: Quadratic Model & Marginal Effect (L10 + L6)

**Model:** $\text{return}_t = \beta_0 + \beta_1 \cdot \text{return}_{t-1} + \beta_2 \cdot \text{return}^2_{t-1} + u_t$

### Quadratic OLS Output

| Variable | Coef | Std Err | t | P>\|t\| |
|---|---|---|---|---|
| Intercept | 0.2255 | 0.0872 | 2.586 | 0.010 |
| return_1 | 0.0486 | 0.0387 | 1.254 | 0.210 |
| return_1_sq | -0.0097 | 0.0070 | -1.385 | 0.167 |

### (i) Marginal Effect when return_{t-1} = 0.10

$$\frac{\partial \text{return}_t}{\partial \text{return}_{t-1}} = \beta_1 + 2\beta_2 \cdot \text{return}_{t-1}$$

$$= 0.0486 + 2 \times (-0.0097) \times 0.10 = 0.0486 - 0.00194 \approx \mathbf{0.047}$$

**Interpretation:** When last week's return was 0.10 (10%), a one-unit increase in $\text{return}_{t-1}$ is associated with a 0.047 unit increase in this week's return.

### (ii) Interpretation of Coefficients

**β₁ = 0.0486 (return_1 coefficient):**
- The linear component; positive coefficient suggests mild positive momentum
- However, p = 0.210 → **not statistically significant** at 5%

**β₂ = −0.0097 (return_1_sq coefficient):**
- The quadratic (curvature) component
- The **negative sign** indicates a concave relationship — the positive effect of past returns diminishes and eventually reverses at extremes
- This implies **mean-reversion** at extremes: very high past returns → relatively lower current returns
- However, p = 0.167 → **not statistically significant** at 5%

**Overall:** Neither the linear nor quadratic terms are significant. The quadratic model does not provide strong evidence of nonlinear predictability, broadly consistent with EMH.

---

## Quick Reference: Key Formulas

| Concept | Formula |
|---|---|
| Logistic probability | $p = \exp(X\beta) / (1 + \exp(X\beta))$ |
| Odds ratio from β | $\exp(\beta)$ |
| % change in odds | $|\exp(\beta) - 1| \times 100\%$ |
| Holt-Winters trend update | $b_t = \beta(a_t - a_{t-1}) + (1-\beta)b_{t-1}$ |
| Forecast h steps ahead | $\hat{y}_{t+h} = a_t + h \cdot b_t$ |
| ACF at lag h | $\rho(h) = \text{Cov}(X_t, X_{t+h}) / \text{Var}(X_t)$ |
| Marginal effect (quadratic) | $\partial Y/\partial X = \beta_1 + 2\beta_2 \cdot X$ |
| Marginal effect (interaction) | $\partial Y/\partial X_1 = \beta_1 + \beta_3 \cdot X_2$ |
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Accuracy | (TP + TN) / Total |
| 95% CI contains zero | Fail to reject H₀ (not significant at 5%) |
| p < 0.05 | Reject H₀ (significant at 5%) |
| p < 0.01 | Significant at 1% |
| p < 0.10 | Significant at 10% |

---

*Reference Lectures: L2 (Pandas I), L3 (Python II / matplotlib), L5 (Regression I), L6 (Regression II), L7 (Logistic Regression), L8 (Time Series I), L9 (Time Series II), L10 (Time Series III)*
