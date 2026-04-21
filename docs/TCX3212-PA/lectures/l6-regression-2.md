---
title: "L6: Regression II"
sidebar_label: "L6: Regression II"
sidebar_position: 6
---

## Statistical Inference in Regression

After fitting a regression model, two key inferential questions arise:

1. **Overall model significance (F-test):** Is at least one predictor $X_1, \ldots, X_p$ useful in predicting $Y$?
2. **Individual predictor significance (t-tests):** Is there a statistically significant association between each specific predictor $X_j$ and $Y$?

---

## Hypothesis Tests and Confidence Intervals for Coefficients

### Standard Errors of Coefficient Estimates

The OLS estimate $\hat{\beta}_1$ is an **unbiased** estimate of the true population slope $\beta_1$. However, estimates vary across different samples. The **standard error** quantifies this variability:

$$\text{SE}(\hat{\beta}_1) = \sqrt{\text{Var}(\hat{\beta}_1)}$$

For the Advertising example:
- $\hat{\beta}_0 = 7.0326$, $\text{SE}(\hat{\beta}_0) = 0.458$
- $\hat{\beta}_1 = 0.0475$, $\text{SE}(\hat{\beta}_1) = 0.003$

$\hat{\beta}_1$ will be closer to the true $\beta_1$ when (1) the sample size $n$ is larger, or (2) the predictor values $x_i$ are more spread out.

### Confidence Intervals for Coefficients

The $100(1-\alpha)\%$ confidence interval for $\beta_j$ is:

$$\hat{\beta}_j \pm t_{(n-p-1,\, \alpha/2)} \cdot \text{SE}(\hat{\beta}_j)$$

where $t_{(n-p-1,\, \alpha/2)}$ is the critical t-value with $n - p - 1$ degrees of freedom.

**Interpretation (95% CI):** With repeated sampling, 95% of such constructed intervals will contain the true unknown $\beta_j$.

For the Advertising data:
- 95% CI for $\beta_0$: $[6.130, 7.935]$ → we are 95% confident that when TV spending is \$0, average sales are between 6,130 and 7,935 units.
- 95% CI for $\beta_1$ (TV): $[0.042, 0.053]$ → we are 95% confident that each additional \$1,000 in TV spending is associated with 42–53 additional units sold.

Since 0 is **not** in the CI for $\beta_1$, TV advertising is **statistically significant**.

### t-Test for Individual Predictors

Tests whether a single predictor $X_j$ is associated with $Y$:

$$H_0: \beta_j = 0 \qquad H_1: \beta_j \neq 0$$

**t-statistic:**

$$t = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}$$

**Decision rule:** If p-value $< \alpha = 0.05$, reject $H_0$ → predictor is statistically significant at 5%.

For TV advertising: p-value $\approx 0.000 < 0.05$ → reject $H_0$ → statistically significant relationship.

> **Note:** A p-value is the probability of observing a t-statistic at least as extreme as the one calculated, *assuming $H_0$ is true*. It is not the probability that $H_0$ is true.

---

## F-Test on Multiple Regression Coefficients

When there are $p$ predictors, use the **F-test** to evaluate overall model significance:

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_k = 0 \qquad H_1: \text{at least one } \beta_j \neq 0$$

The F-statistic compares the model's explained variance to its unexplained variance:

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)}$$

- Under $H_0$, $F \approx 1$.
- A large $F$ (and small p-value) → reject $H_0$ → at least one predictor is useful.

**Example:** For the Advertising dataset with TV, Radio, Newspaper: F-test p-value $= 1.58 \times 10^{-96} < 0.05$ → strong evidence that at least one medium is associated with sales.

> **Why not just check individual t-tests?** With many predictors, even if all $\beta_j = 0$, some t-tests will appear significant by chance (multiple testing problem). The F-test controls for this.

---

## Coding Scheme for Categorical Variables

Regression requires **numerical** inputs. Qualitative (categorical) predictors must be encoded as **dummy (indicator) variables**.

### Two-Level Categorical Variable (e.g., Gender)

Create one binary indicator:

$$x_i = \begin{cases} 1 & \text{if female} \\ 0 & \text{if male (reference)} \end{cases}$$

Model:
$$\text{Balance} = \beta_0 + \beta_1 \cdot \text{Income} + \beta_2 \cdot x_{\text{female}} + \varepsilon$$

Interpretation:
- $\hat{\beta}_0 = 233.77$: Predicted average balance for a **male** with income = \$0.
- $\hat{\beta}_1 = 6.05$: For each one-unit increase in income, average balance increases by \$6.05, holding Gender constant.
- $\hat{\beta}_2 = 24.31$: Females carry on average \$24.31 more in credit card balance than males with the **same income level**.

The choice of reference level (male = 0 vs female = 0) does not affect model fit — only the interpretation of the intercept and dummy coefficient.

```python
import statsmodels.formula.api as smf

# C() tells Python to treat Gender as categorical; Treatment sets the reference level
model = smf.ols('Balance ~ Income + C(Gender, Treatment(reference="Male"))',
                data=credit).fit()
```

### Three-or-More Level Categorical Variable (e.g., Ethnicity)

For $k$ categories, create **$k - 1$ dummy variables**. The omitted category is the **reference group**.

Example: Ethnicity ∈ {African American, Asian, Caucasian}

| Person | Asian | Caucasian | Ethnicity |
|--------|-------|-----------|-----------|
| A | 0 | 0 | African American (reference) |
| B | 1 | 0 | Asian |
| C | 0 | 1 | Caucasian |

Interpretation:
- $\hat{\beta}_1$ = average difference in balance: **Asian vs African American**
- $\hat{\beta}_2$ = average difference in balance: **Caucasian vs African American**
- $\hat{\beta}_2 - \hat{\beta}_1$ = average difference in balance: **Caucasian vs Asian**

To test if Caucasian and African American differ: $H_0: \beta_2 = 0$.

---

## Interaction Terms in Regression

**Additive model** (no interaction): assumes the effect of one predictor is the same regardless of another predictor's value.

$$\text{Balance} = \beta_0 + \beta_1 \cdot \text{Income} + \beta_2 \cdot \text{Student} + \varepsilon$$

Both students and non-students share the **same slope** $\beta_1$ for Income. They only differ in their intercepts.

**Interaction model**: allows the effect of Income on Balance to depend on student status.

$$\text{Balance} \approx \beta_0 + \beta_1 \cdot \text{Income} + \beta_2 \cdot \text{Student} + \beta_3 \cdot (\text{Income} \times \text{Student}) + \varepsilon$$

Expanding:
$$\text{Balance} = \begin{cases} (\beta_0 + \beta_2) + (\beta_1 + \beta_3) \cdot \text{Income} & \text{if Student} \\ \beta_0 + \beta_1 \cdot \text{Income} & \text{if Not Student} \end{cases}$$

- Effect of Income on Balance for **students**: $\beta_1 + \beta_3$
- Effect of Income on Balance for **non-students**: $\beta_1$
- $\beta_3$: how much the income slope *differs* for students vs non-students.

**Marginal effect rule:** For any non-linear or interaction model, the marginal effect of $X_k$ on $Y$ is the partial derivative $\frac{\partial Y}{\partial X_k}$.

```python
# Interaction term in statsmodels
smf.ols('Balance ~ Income * C(Student, Treatment(reference="No"))',
        data=df).fit()
```

> **Interpretation of interaction:** If $\hat{\beta}_3 = -1.99$ and $\hat{\beta}_1 = 6.22$, the income slope for students is $6.22 + (-1.99) = 4.23$ — students see a smaller boost in balance from each additional unit of income compared to non-students. (Even if the interaction term is not statistically significant, it is the correct way to compute the student slope.)

---

## Practice Questions

**Q1.** The 95% confidence interval for the TV coefficient is $[0.042, 0.053]$ (TV measured in \$1,000s). Which statement is correct?

- **A) ✓** We are 95% confident that every additional \$1,000 in TV advertising is associated with a 42 to 53 unit increase in sales on average.
- B) There is a 95% probability that a \$1,000 increase in TV advertising will increase sales by 42–53 units.
- C) Because the interval excludes zero, TV advertising is not statistically significant.

> **Why A:** A confidence interval is a statement about the *parameter* (over repeated samples), not a probability about any single observation. And excluding zero means TV advertising *is* significant, not the opposite.

---

**Q2.** In the model `Balance ~ Income + Gender` with Male as reference, $\hat{\beta}_0 = 233.77$. What does this represent?

- A) The predicted average balance for a Female with income = \$0.
- **B) ✓** The predicted average balance for a **Male** with income = \$0.
- C) The average difference in balance between females and males.
- D) Females carry \$233.77 more credit debt than males.

> **Why B:** $\hat{\beta}_0$ is the intercept, interpreted at the reference level of all categorical variables (Male) and the baseline of all quantitative variables (Income = 0).

---

**Q3.** To test whether at least one of the advertising media is useful for predicting sales, which hypotheses are correct for an F-test?

- **A) ✓** $H_0: \beta_{\text{TV}} = \beta_{\text{Radio}} = \beta_{\text{Newspaper}} = 0$; $H_1:$ at least one $\beta_j \neq 0$.
- B) $H_0: \beta_{\text{TV}} = 0$; $H_1: \beta_{\text{TV}} \neq 0$. (This is a t-test, not F.)
- C) $H_0: \beta_{\text{intercept}} = \beta_{\text{TV}} = \beta_{\text{Radio}} = \beta_{\text{Newspaper}} = 0$; $H_1:$ more than one $\beta_j \neq 0$.

> **Why A:** The F-test tests whether all *slope* coefficients are simultaneously zero. The intercept is not included in the null hypothesis.

---

**Q4.** In the interaction model `Balance ~ Income + Student + Income*Student`, what is the marginal effect of income on balance for a student?

- A) $b_1 = 6.22$ (the interaction term is not significant).
- **B) ✓** $b_1 + b_3 = 6.22 + (-1.99) = 4.23$ (income slope plus the student interaction).
- C) $b_3 = -1.99$ (only the interaction term).

> **Why B:** For students (Student = 1), the model becomes Balance = $(\beta_0 + \beta_2) + (\beta_1 + \beta_3) \cdot \text{Income}$. The slope of Income for students is $\beta_1 + \beta_3$, regardless of statistical significance of $\beta_3$.
