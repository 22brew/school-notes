---
title: "L8: Time Series I"
sidebar_label: "L8: Time Series I"
sidebar_position: 8
---

## Introduction to Time Series Data

### Cross-Sectional vs Time Series Data

**Cross-sectional data** is a snapshot taken at a single point in time (e.g., a salary survey for 2023, a national travel survey). Each observation is a different unit (person, firm, household) sampled from a population.

**Time series data** consists of observations on one or more variables taken *sequentially over time*:

$$\{(t, x_t) : t = 1, 2, \ldots, n\}$$

Examples:
- Daily high temperature in a city: $t$ = calendar day, $x_t$ = temperature.
- Quarterly earnings per share: $t$ = fiscal quarter, $x_t$ = EPS.
- Monthly airline passenger counts.

The key distinction: in time series, observations are **ordered** and often **correlated** — knowing the past tells you something about the future.

### Time Series vs Time Series Model

- A **time series** is just the observed data: $(t, x_t)$ pairs.
- A **time series model** is a stochastic process — a probability model governing $X_t$ (a random variable) for each $t$. The observed data is one possible realisation of this process.

The modelling paradigm:
1. Observe time series data.
2. Choose an appropriate family of models.
3. Fit the model.
4. Use it to forecast.

---

## Components of a Time Series

Any time series can be decomposed into four components:

| Component | Description |
|-----------|-------------|
| **Trend ($T_t$)** | Long-run upward or downward movement over many periods. |
| **Cycle ($C_t$)** | Recurring up/down movements around the trend, typically over multiple years (e.g., business cycles). Often merged with trend. |
| **Seasonality ($S_t$)** | Periodic patterns completing within a calendar year (monthly, quarterly). Repeat annually. |
| **Irregular ($R_t$)** | Random "leftover" variation after the other components are removed. |

### Additive vs Multiplicative Decomposition

**Additive model** (use when seasonal variation is roughly constant in magnitude):
$$Y_t = T_t + C_t + S_t + R_t$$

**Multiplicative model** (use when seasonal variation grows with the level of the series):
$$Y_t = T_t \times C_t \times S_t \times R_t$$

*Diagnostic:* If the residuals from an additive decomposition still show a repeating seasonal pattern, or if the variance of the original series grows over time, switch to the multiplicative model.

```python
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

df = sm.datasets.get_rdataset("AirPassengers").data
df.index = pd.date_range(start='1949-01-01', periods=len(df), freq='ME')
series = df['value']

# Multiplicative decomposition (variance grows over time)
result = seasonal_decompose(series, model='multiplicative')
result.plot()
plt.show()
```

---

## Stationarity

### Why Stationarity Matters

Most time series models (including ARIMA, covered later) assume the data is **stationary**. Non-stationary processes have statistics that change over time, making it impossible to learn stable patterns.

With only one observation per time point, we cannot infer the distribution of $X_t$ from $x_t$ alone. **Stationarity** is the assumption that allows us to pool information across time — if the process is time-shift invariant, observations at different times share the same statistical properties.

### Weak (Covariance) Stationarity

A process $(X_t)$ is **weakly stationary** if, for all $t$:

1. $\mathbb{E}[X_t] = \mu$ (constant mean, does not depend on $t$)
2. $\text{Var}(X_t) = \sigma^2$ (constant variance)
3. $\text{Cov}(X_t, X_{t+h}) = \gamma(h)$ (autocovariance depends only on lag $h$, not on $t$)

Weak stationarity is sufficient for most time series methods. When "stationary" is mentioned in this course without qualification, it means weak stationarity.

### Ways Stationarity Can Fail

| Failure Mode | Example | Fix |
|-------------|---------|-----|
| Trend | Rising mean over time | Detrending or differencing |
| Seasonality | Repeating annual pattern | Seasonal differencing |
| Changing variance | Variance grows with level | Log or Box-Cox transformation |
| Structural break | Mean/variance shifts abruptly | Split the series |

### Strong vs Weak Stationarity

**Strong stationarity:** The entire joint distribution of $(X_1, \ldots, X_n)$ is invariant to time shifts. Very restrictive; rarely verified in practice.

**Weak stationarity:** Only requires constant mean, constant variance, and lag-dependent autocovariance. Much more practical.

---

## Autocorrelation Function (ACF)

### Autocovariance Function (ACVF)

For a stationary process, the **autocovariance function (ACVF)** at lag $h$ is:

$$\gamma_X(h) = \text{Cov}(X_t, X_{t+h})$$

It measures how $X_t$ and $X_{t+h}$ move together — the linear dependence of the series on its own past values.

### Autocorrelation Function (ACF)

The **ACF** normalises the ACVF:

$$\rho_X(h) = \text{corr}(X_t, X_{t+h}) = \frac{\gamma_X(h)}{\gamma_X(0)}$$

- $\rho_X(0) = 1$ always (a variable is perfectly correlated with itself).
- $\rho_X(h) \in [-1, 1]$.

The **correlogram** is a plot of sample ACF values against lag $h$.

### Interpreting the Correlogram

**White noise** ($W_t \sim \text{i.i.d.}(0, \sigma^2)$) — the simplest stationary process:
- $\rho_X(h) = 0$ for all $h \neq 0$.
- On the correlogram, ~95% of sample ACF values should fall within the dashed blue lines: $\pm 1.96/\sqrt{n}$.
- Any bar exceeding the dashed lines suggests statistically significant autocorrelation at that lag.

**MA(1) process** ($X_t = W_t + \theta W_{t-1}$, $\theta = 0.7$):
- Correlation at lag 1 is significant (≈ 0.47).
- No significant correlation beyond lag 1 — the "cut-off" after lag 1 is a hallmark of an MA(q) process.

> **ACF interpretation rule:** A "cut-off" at lag $q$ (significant up to lag $q$, then drops to zero) suggests an **MA(q)** component. A slow geometric decay suggests an **AR** process.

---

## Forecasting Methods

### Naïve Method

Set all forecasts equal to the most recent observed value:

$$\hat{y}_{T+h} = y_T \qquad \text{for } h = 1, 2, 3, \ldots$$

Simple but effective for stationary data or as a baseline benchmark.

### Seasonal Naïve Method

Set each forecast equal to the last observed value from the **same season**:

$$\hat{y}_{T+h} = y_{T+h - m(k+1)}$$

where $m$ is the seasonal period and $k$ is the number of complete years in the forecast period before time $T+h$.

*Example:* To forecast January 2026, use the value from January 2025.

### Centered Moving Average (CMA)

Used for **smoothing** (not forecasting) to expose the underlying trend by eliminating seasonal variation:

$$\text{CMA}_t = \frac{y_{t-(w-1)/2} + \cdots + y_t + \cdots + y_{t+(w-1)/2}}{w}$$

where $w$ is the window size (choose $w = 12$ for monthly data, $w = 4$ for quarterly data to capture a full cycle).

Note: No CMA value at the first or last $(w-1)/2$ time points (insufficient neighbours).

```python
# Centered 12-period moving average
ma_centered = series.rolling(12, center=True).mean()
```

### Simple Moving Average (SMA) Forecasting

Use the average of the $w$ most recent observations as the forecast for the next period:

$$\hat{y}_{T+h} = \frac{y_T + y_{T-1} + \cdots + y_{T-w+1}}{w}$$

**Tradeoff:** Large $w$ smooths noise but reacts slowly to changes; small $w$ is more responsive but noisier.

```python
# 3-period SMA
sma = series.rolling(3).mean()
forecast = sma.shift(1)  # use current SMA to forecast next period
```

| Window $w$ | Characteristic |
|------------|---------------|
| Small (e.g., 3) | Responsive; choppy forecasts |
| Medium (e.g., 5) | Smoother; some lag |
| Large (e.g., 12) | Very smooth; lags behind turning points |

> SMA works best for **stationary** series (no trend or seasonality). Stationarity ensures that recent history is representative of the future.

---

## Model Evaluation: Data Partitioning

Divide time series data chronologically:

- **Training set (in-sample):** Used to fit the model.
- **Test set (out-of-sample):** Used to evaluate forecast accuracy on unseen future periods.

> ⚠️ Never use future data to train a time series model — this is "look-ahead bias" and gives falsely optimistic performance estimates.

Forecast accuracy on the test set tells you how well the model generalises to the future.
