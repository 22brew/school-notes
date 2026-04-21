---
title: "L9: Time Series II"
sidebar_label: "L9: Time Series II"
sidebar_position: 9
---

## Exponential Smoothing Models

Exponential smoothing methods improve on Simple Moving Average (SMA) by giving **more weight to recent observations** and exponentially less weight to older ones. Unlike SMA (which weights all window observations equally), exponential smoothing uses all past data with geometrically decaying weights.

There are three variants depending on the structure of the data:

| Method | Handles Trend | Handles Seasonality | Parameters |
|--------|--------------|-------------------|------------|
| Simple Exponential Smoothing (SES) | No | No | $\alpha$ |
| Double (Holt's) | Yes | No | $\alpha$, $\beta$ |
| Triple (Holt-Winters) | Yes | Yes | $\alpha$, $\beta$, $\gamma$ |

---

## Simple Exponential Smoothing (SES)

**Use case:** Time series with a changing mean but **no trend and no seasonality**.

### Smoothing Equation

The smoothed level $a_t$ at time $t$ is:

$$a_t = \alpha \cdot y_t + (1 - \alpha) \cdot a_{t-1}, \qquad 0 \leq \alpha \leq 1$$

Expanding recursively:

$$a_t = \alpha \cdot y_t + \alpha(1-\alpha) \cdot y_{t-1} + \alpha(1-\alpha)^2 \cdot y_{t-2} + \cdots$$

Each past observation gets weight $\alpha(1-\alpha)^k$ — **exponentially decaying** with lag $k$.

- **Large $\alpha$** (close to 1): more weight to recent observations → model adapts quickly, noisier.
- **Small $\alpha$** (close to 0): more weight to history → smoother, slower to adapt.

Initialisation: $a_1 = y_1$.

### Forecasting

SES produces a **"flat" forecast** — all future forecasts equal the last smoothed level:

$$\hat{y}_{T+h} = a_T \qquad \text{for all } h = 1, 2, 3, \ldots$$

One-step-ahead fitted values: $\hat{y}_{t+1} = a_t$.

> **Note:** SES is equivalent to an ARIMA(0,1,1) model (one nonseasonal difference, MA(1) term, no constant).

### Prediction Intervals

A 95% prediction interval for $\hat{y}_{T+h}$:

$$\hat{y}_{T+h} \pm 1.96 \cdot s \cdot \sqrt{1 + (h-1)\alpha^2}$$

where $s$ is the standard error of residuals. Intervals widen for longer horizons.

### Python Example

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pandas as pd

oil = pd.read_csv("Oil.csv", index_col='Year', parse_dates=True)
series = oil['Production']

# Fit SES with alpha = 0.83
model = SimpleExpSmoothing(series).fit(smoothing_level=0.83, optimized=False)
forecast = model.forecast(5)    # 5-step ahead forecast
model.fittedvalues              # in-sample predictions
```

---

## Double Exponential Smoothing (Holt's Method)

**Use case:** Time series that exhibits a **linear trend** but no seasonality.

### Equations

$$\hat{y}_{t+k} = a_t + b_t \cdot k$$

where $a_t$ (level) and $b_t$ (trend slope) are updated at each step:

$$a_t = \alpha \cdot y_t + (1 - \alpha)(a_{t-1} + b_{t-1})$$
$$b_t = \beta(a_t - a_{t-1}) + (1 - \beta) b_{t-1}$$

- $\alpha$: smoothing constant for the level.
- $\beta$: smoothing constant for the trend.
- **Small $\beta$**: trend changes slowly; the slope is relatively stable.
- **Large $\beta$**: trend adapts quickly to recent data.

Initialise: $a_1 = y_1$, $b_1 = y_2 - y_1$.

### Forecasting

$$\hat{y}_{T+h} = a_T + h \cdot b_T$$

Unlike SES, the forecast function is **trending** — it projects the current level with the current slope $h$ steps forward.

### Example (Australian Air Passengers)

Smoothing parameters: $\alpha = 0.8321$, $\beta = 0.0001$ (nearly constant slope).

Given $a_1 = 15.57$, $b_1 = 2.102$:

$$\hat{y}_2 = a_1 + 1 \cdot b_1 = 15.57 + 2.102 = 17.67$$

Then update:

$$a_2 = \alpha \cdot y_2 + (1-\alpha)(a_1 + b_1)$$
$$b_2 = \beta(a_2 - a_1) + (1-\beta) b_1$$

---

## Triple Exponential Smoothing (Holt-Winters)

**Use case:** Time series with **both trend and seasonality**. $L$ = number of seasons per year ($L = 12$ monthly, $L = 4$ quarterly).

### Additive Holt-Winters

For **constant seasonal amplitude** (additive seasonality):

$$\hat{y}_{t+k} = a_t + b_t \cdot k + c_{t+k-L}$$

**Update equations:**

$$a_t = \alpha(y_t - c_{t-L}) + (1 - \alpha)(a_{t-1} + b_{t-1}) \qquad \text{(seasonally adjusted level)}$$
$$b_t = \beta(a_t - a_{t-1}) + (1 - \beta) b_{t-1} \qquad \text{(trend)}$$
$$c_t = \gamma(y_t - a_t) + (1 - \gamma) c_{t-L} \qquad \text{(seasonal index)}$$

- $c_{t-L}$: seasonal factor from the same season one year ago.
- Seasonal factor $c_t$ is the additive deviation from the (de-seasoned) level.

### Multiplicative Holt-Winters

For **growing seasonal amplitude** (multiplicative seasonality):

$$\hat{y}_{t+k} = (a_t + b_t \cdot k) \times c_{t+k-L}$$

$$a_t = \alpha(y_t / c_{t-L}) + (1 - \alpha)(a_{t-1} + b_{t-1})$$
$$b_t = \beta(a_t - a_{t-1}) + (1 - \beta) b_{t-1}$$
$$c_t = \gamma(y_t / a_t) + (1 - \gamma) c_{t-L}$$

Here $c_t$ is a **multiplicative** factor (e.g., 1.24 means that season is 24% above the de-seasoned level).

### Python (Holt-Winters)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Multiplicative Holt-Winters (monthly data, L=12)
hw_model = ExponentialSmoothing(
    series,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
)
fit = hw_model.fit(
    smoothing_level=0.441,
    smoothing_trend=0.03,
    smoothing_seasonal=0.002,
    optimized=False
)
forecast = fit.forecast(8)
```

---

## Performance Evaluation Metrics

The forecast error at time $t$:

$$e_t = y_t - \hat{y}_t$$

### Mean Absolute Deviation (MAD) / Mean Absolute Error (MAE)

$$\text{MAD} = \frac{1}{n}\sum_{t=1}^n |y_t - \hat{y}_t|$$

- Robust to extreme errors; easy to interpret.
- Unit: same as the original series.

### Mean Squared Error (MSE) and RMSE

$$\text{MSE} = \frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}$$

- Penalises large errors more heavily (due to squaring).
- RMSE is on the same scale as the original series.

### Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{n}\sum_{t=1}^n \left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%$$

- Scale-free; useful for comparing across different series.
- Undefined when $y_t = 0$; biased when values are small.

**Example:**

| $y_t$ | $\hat{y}_t$ | $e_t$ | $|e_t|$ | $\text{APE}_t$ |
|-------|------------|-------|---------|----------------|
| 25 | 22 | 3 | 3 | 12.0% |
| 28 | 30 | −2 | 2 | 7.1% |
| 29 | 30 | −1 | 1 | 3.5% |

- MAD = (3+2+1)/3 = 2
- MAPE = (12+7.1+3.5)/3 = 7.5%

| Metric | Sensitive to outliers? | Scale-dependent? |
|--------|----------------------|-----------------|
| MAD | No | Yes |
| MSE | Yes (squares) | Yes |
| RMSE | Yes | Yes |
| MAPE | No | No |

---

## Time Series Differencing

### Recall: Stationarity

Most time series models require stationarity. Non-stationary series (with trends, seasonality, or changing variance) must be transformed first.

Common transformations:
- **Differencing** — removes trends.
- **Log transformation** — stabilises variance.
- **Detrending** — explicitly removes linear trend.

### First-Order Differencing

Subtract the previous observation:

$$y'_t = y_t - y_{t-1}, \qquad t = 2, 3, \ldots, n$$

The differenced series has $n - 1$ values. This removes a **linear trend** — if the original series grows roughly by a constant amount each period, the differenced series will be approximately stationary.

### Second-Order Differencing

Apply differencing twice (if first-order differencing is insufficient):

$$y''_t = y'_t - y'_{t-1}$$

The twice-differenced series has $n - 2$ values. Removes **quadratic trends**. In practice, rarely need to go beyond second-order.

### Seasonal Differencing

Subtract the observation from the same season in the previous cycle:

$$y'_t = y_t - y_{t-m}$$

where $m$ is the seasonal period (12 for monthly, 4 for quarterly). Also called "lag-$m$ differencing."

```python
# First-order differencing
series_diff = series.diff()

# Seasonal differencing (monthly data)
series_seasonal_diff = series.diff(12)
```

### Example: Global Temperature Anomalies

Original series: non-stationary (clear upward trend).

After first-order differencing ($y'_t = y_t - y_{t-1}$): approximately stationary — mean and variance are roughly constant over time.

> **Connection to ARIMA:** The "I" in ARIMA(p, **d**, q) is the order of differencing needed to achieve stationarity. Examining the ACF of the differenced series helps identify the MA order $q$; the PACF helps identify the AR order $p$.
