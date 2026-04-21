---
title: "L10: Time Series III"
sidebar_label: "L10: Time Series III"
sidebar_position: 10
---

## Regression-Based Time Series Forecasting

In general, a time series model for $Y_t$ can include:
- **Lagged values of $Y$** (autoregressive terms): $Y_{t-1}, Y_{t-2}, \ldots$
- **Current and lagged values of other variables** $X$: regressors indexed by time.

$$Y_t = b_0 + b_1 Y_{t-1} + b_2 X_t + \cdots$$

This lecture focuses on regression-based approaches where external predictors — possibly offset in time — help explain $Y_t$.

---

## Static Models

A **static model** models a contemporaneous (same time period) relationship between $Y$ and one or more $X$ variables:

$$y_t = \beta_0 + \beta_1 x_{1t} + \beta_2 x_{2t} + \cdots + \beta_k x_{kt} + \varepsilon_t, \qquad t = 1, 2, \ldots, n$$

"Contemporaneous" means both $y$ and all $x$ variables are measured at the **same time period $t$**.

**Key interpretation:** A change in $x_{kt}$ at time $t$ has an **immediate** effect on $y_t$:

$$\Delta y_t = \beta_k \Delta x_{kt} \qquad \text{(when } \Delta \varepsilon_t = 0\text{)}$$

### OLS Assumptions for Time Series

For OLS to be valid in a time series setting:

| Assumption | Meaning |
|------------|---------|
| TS1: Mean-Zero Error | $\mathbb{E}[\varepsilon_t \mid X_t] = 0$ |
| TS2: Homoskedasticity | $\text{Var}(\varepsilon_t \mid X_t) = \sigma^2$ (constant) |
| TS3: Uncorrelated Errors | $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ for $t \neq s$ |
| TS4: Weak Dependence | $\{(X_t, Y_t)\}$ is stationary and weakly dependent |
| TS5: Linearity | Model is correctly specified as linear |

When TS1–TS5 hold, OLS estimators work the same as in cross-sectional data (unbiased, efficient, standard inference valid).

### Example 1: Phillips Curve (Inflation vs Unemployment)

$$\text{inf}_t = \beta_0 + \beta_1 \text{unemp}_t + \varepsilon_t$$

Study the contemporaneous relationship between US annual inflation rate and unemployment rate (1948–2003). This is a classic static model in macroeconomics — the Phillips Curve hypothesis.

### Example 2: Interest Rates

$$\hat{i3}_t = 1.73 + 0.606 \cdot \text{inf}_t + 0.513 \cdot \text{def}_t$$

where $i3$ = 3-month T-bill rate, $\text{inf}$ = annual CPI inflation rate, $\text{def}$ = government deficit as % of GDP.

**Interpretation:**
- **Intercept (1.73):** If both inflation and deficit are zero, the baseline T-bill rate is ~1.73%.
- **Inflation coefficient (0.606):** A 1 percentage-point increase in inflation is associated with a 0.606 point increase in the T-bill rate, holding the deficit constant. (p-value < 0.05 → significant.)
- **Deficit coefficient (0.513):** A 1 percentage-point increase in the deficit/GDP ratio is associated with a 0.513 point increase in the T-bill rate. (p-value < 0.05 → significant.)

### Example 3: Fertility Rate with Dummy Variables

$$\widehat{\text{gfr}}_t = 98.68 + 0.083 \cdot \text{pe}_t - 24.24 \cdot \text{ww2}_t - 31.59 \cdot \text{pill}_t$$

where:
- $\text{gfr}$ = general fertility rate (children per 1,000 women)
- $\text{pe}$ = personal tax exemption (\$)
- $\text{ww2}$ = 1 during World War II (1941–1945), 0 otherwise
- $\text{pill}$ = 1 after 1963 (availability of oral contraceptive pill), 0 otherwise

**Interpretation:**
- **ww2 (−24.24):** Holding other variables constant, fertility was ~24 fewer births per 1,000 women during WWII compared to peacetime.
- **pill (−31.59):** After 1963, fertility was ~31.6 fewer births per 1,000 women per year, controlling for tax exemptions and wartime status. The pill dummy is highly significant (p < 0.05).

---

## Time Series with Trend

### The Problem: Spurious Regression

When two variables both have **upward trends**, regressing one on the other can produce a spuriously significant coefficient — even if the variables have no genuine causal relationship. Including a **time trend variable** $t$ in the regression controls for this.

### Linear Time Trend Model

$$y_t = \beta_0 + \beta_1 t + \varepsilon_t$$

- $E(y_t) = \beta_0 + \beta_1 t$ is a linear function of time.
- $\beta_1 > 0$: on average, $y_t$ has an upward trend of $\beta_1$ per period.
- $\beta_1 < 0$: downward trend.
- First difference: $y_t - y_{t-1} = \beta_1$ (constant), so linear trend → constant increment.

### Including Trend as a Covariate

Add $t$ as a predictor when $y_t$ and one of the $x$ variables share a common trend:

$$y_t = \beta_0 + \beta_1 x_{1t} + \beta_2 x_{2t} + \beta_3 t + \varepsilon_t$$

Omitting the trend when it's present causes **omitted variable bias** — the $x$ coefficients absorb the trend, making them biased and misleading.

### Example: Housing Investment and Prices

**Without trend:**
$$\widehat{\log(\text{invpc})}_t = -0.550 + 1.241 \log(\text{price})_t$$
Appears to show a strong positive relationship — but both variables trend upward over time. This is a **spurious** relationship driven by the common trend.

**With trend:**
$$\widehat{\log(\text{invpc})}_t = -0.913 - 0.381 \log(\text{price})_t + 0.0098 t$$
- The price elasticity is now negative and **not statistically significant**.
- The trend is significant: per-capita housing investment grows by ~0.98% (~1%) per year, independent of price.
- Conclusion: there is no evidence that price drives housing investment — the apparent relationship was entirely due to both variables trending upward together.

---

## Time Series with Seasonality

When data exhibits periodic seasonal patterns (monthly, quarterly), include **seasonal dummy variables**.

For monthly data with January as the base month, add 11 dummies (February through December):

$$y_t = \beta_0 + \beta_1 x_{1t} + \cdots + \delta_1 \text{Feb}_t + \delta_2 \text{Mar}_t + \cdots + \delta_{11} \text{Dec}_t + \varepsilon_t$$

- $\beta_0$ = intercept for January (base month).
- $\delta_k$ = the average difference in $y$ during month $k$ compared to January, holding other variables constant.

**General rule:** For $L$ seasons (months/quarters), create **$L - 1$ seasonal dummies**. The omitted season is the reference.

For quarterly data:
$$y_t = \beta_0 + \beta_1 x_t + \delta_1 Q2_t + \delta_2 Q3_t + \delta_3 Q4_t + \varepsilon_t$$

Seasonal dummies and the time trend can be combined to capture both components simultaneously.

---

## Leading and Lagging Variables

Sometimes $x_t$ and $y_t$ are related but **offset in time**:

- **Leading variable:** Changes in $x$ *precede* changes in $y$. ($x$ leads $y$.)
- **Lagging variable:** Changes in $x$ *follow* changes in $y$. ($x$ lags $y$.)
- **Coincident:** Both move together in the same period.

**How to identify:** Draw a vertical line near a prominent feature on a time series plot. Whichever variable "reaches" the landmark first is the leading variable.

**Examples:**

| Leading Variable | Lagging Variable | Lag |
|-----------------|-----------------|-----|
| Haze level | Mask sales | A few days |
| Forest fires in Indonesia | Haze in Singapore | A few days |
| Cigarette tax increase | Cigarette consumption | A few months |
| Study hours | Exam grade | Days |

---

## Finite Distributed Lag (FDL) Models

When $x$ is a **leading variable** for $y$, we can model the **delayed effects** of $x$ on $y$ explicitly:

$$y_t = \beta_0 + \beta_1 x_t + \beta_2 x_{t-1} + \beta_3 x_{t-2} + \varepsilon_t$$

### Coefficient Interpretation

| Coefficient | Name | Interpretation |
|------------|------|---------------|
| $\beta_1$ | **Impact effect** | Immediate effect of a 1-unit change in $x_t$ on $y_t$, holding all past $x$ constant. |
| $\beta_2$ | **1-period dynamic multiplier** | Effect of $x_{t-1}$ on $y_t$ — the response one period after the shock. |
| $\beta_3$ | **2-period dynamic multiplier** | Effect of $x_{t-2}$ on $y_t$ — the response two periods after the shock. |
| $\beta_1 + \beta_2 + \beta_3$ | **Long-run (cumulative) multiplier** | Total effect on $y$ of a permanent 1-unit increase in $x$, after all lags have worked through. |

### Example: Haze and Mask Sales

Suppose people respond to haze with a 1-day lag (they need time to go buy masks):

$$\text{MaskSales}_t = b_0 + b_1 \cdot \text{Haze}_{t-1} + b_2 \cdot \text{Mother} + b_3 \cdot \text{Age} + \varepsilon_t$$

Additional control variables (Mother, Age) are **not** time-indexed — they are treated as cross-sectional covariates and interpreted the same as in standard multiple regression.

### Autoregressive Model

A special case of lagged regression using lagged values of $Y$ itself:

$$\text{Sales}_t = b_0 + b_1 \cdot \text{Sales}_{t-1} + \varepsilon_t$$

This is called an **AR(1)** (autoregressive of order 1) model. Key difference from SMA:
- SMA takes an unweighted average of the last $w$ observations.
- AR(1) estimates a *regression coefficient* $b_1$ that is learned from the data — not necessarily $1/w$.

---

## Summary

| Model Type | When to Use | Key Features |
|-----------|-------------|-------------|
| **Static Model** | Contemporaneous relationship between $y$ and $x$ | Same-period effects; standard OLS assumptions |
| **Trend Model** | $y_t$ has a linear trend (or $x$ and $y$ share a trend) | Add $t$ as a covariate to avoid spurious regression |
| **Seasonal Model** | Regular periodic patterns in $y$ | Add $L-1$ seasonal dummy variables |
| **FDL Model** | $x$ is a leading variable; delayed effects on $y$ | Include $x_{t-1}, x_{t-2}, \ldots$; interpret cumulative multiplier |
| **AR Model** | $y$ depends on its own past values | Lagged $Y$ as predictor; basis for ARIMA |
