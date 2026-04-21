---
title: "L3: Python Basics II"
sidebar_label: "L3: Python Basics II"
sidebar_position: 3
---

## Data Visualisation in Python

Effective visualisation translates raw numbers into human-readable insight. The Python visualisation ecosystem has two main layers:

- **Matplotlib** — the low-level foundation. Full control, more verbose.
- **Seaborn** — high-level API built on Matplotlib. Best for tidy (long-form) data and statistical plots.

> Visualisations are for **humans**: the goal is to communicate patterns clearly, not just to render pixels.

---

## Matplotlib

Matplotlib is the underlying library for Seaborn, Pandas native plotting, and most Python plotting tools.

### Key Concepts

| Object | Description |
|--------|-------------|
| **Figure** | The top-level container (the entire image) |
| **Axes** | One individual chart/plot within a Figure |
| **Axis** | The x-axis or y-axis *within* an Axes object |

> ⚠️ **Axes ≠ Axis.** "Axes" = a single chart; "Axis" = x or y.

Matplotlib remembers the current Axes for the duration of a cell (hidden state). When creating a plot, a new figure + axes is created if none is active.

### Creating Figures and Axes

```python
import matplotlib.pyplot as plt
import numpy as np

# Empty figure
fig = plt.figure()

# 2×2 grid of subplots
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 4), layout="constrained")

# Single axes (most common)
fig, ax = plt.subplots(figsize=(5.4, 5.4), layout='constrained')
```

### Basic Plotting

```python
x = np.arange(100)
y = x

ax.plot(x, y, label="y = x")
```

Format strings combine line style, colour, and marker:

```python
x = np.arange(0, 5, 0.1)
plt.plot(x, x**2, ':bo')   # dotted line, blue, circle markers
# Long form equivalent:
plt.plot(x, x**2, linestyle=':', color='b', marker='o')
```

### Axes Customisation

```python
ax.set(
    title="Manual ticks",
    xlabel="X-axis",
    ylabel="Y-axis",
    yticks=np.linspace(0, 100, 4),
    xticks=np.arange(0.5, 101, 20),
    xticklabels=[f"${v:1.2f}" for v in np.arange(0.5, 101, 20)]
)

ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
```

Line style options: `"-"` solid, `"--"` dashed, `":"` dotted, `"-."` dash-dot.

### Annotations

```python
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
ax.plot(t, s)
ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
```

Annotation coordinates are in data units (not pixels), making them easy to position.

### Saving Figures

```python
plt.savefig("chart.jpg", format='jpg')
plt.savefig("./images/plot.png")  # auto-detect format from extension
```

---

## Common Chart Types

### Histogram

Used for **one quantitative variable** to show its distribution.

```python
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10_000)

plt.hist(x, bins=50, density=True, facecolor='g')
plt.xlabel('Value')
plt.ylabel('Probability density')
plt.title('Histogram')
plt.grid(True)
```

- `density=True` normalises the histogram so the total area = 1.
- Choose `bins` carefully: too few hides structure, too many reveals noise.

### Bar Chart

Used for **categorical data** — height represents the value for each category.

```python
counts = df["Category"].value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel("Category")
plt.ylabel("Count")
```

**Stacked bar chart:**
```python
plt.bar(years, cases, label="Cases")
plt.bar(years, deaths, bottom=cases, label="Deaths")  # stacks on top of Cases
plt.legend()
```

**Clustered (side-by-side) bar chart:**
```python
width = 0.35
plt.bar(x - width/2, cases, width, label='Cases')
plt.bar(x + width/2, deaths, width, label='Deaths')
```

Stacked bars can be misleading when one series is a *subset* of the other. Clustered bars allow direct comparison of each category.

### Box Plot

Visualises **distribution** using quartiles. Useful for comparing distributions across groups.

```python
plt.boxplot(data, vert=True)
```

| Element | Meaning |
|---------|---------|
| Centre line | Median (Q2, 50th percentile) |
| Box edges | Q1 (25th) and Q3 (75th) |
| Whiskers | Extend to min/max *excluding* outliers |
| Outliers | Points beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR |

**Side-by-side box plots:**
```python
# data_by_group is a list of 1D arrays, one per group
plt.boxplot(data_by_group, tick_labels=group_labels, vert=True)
```

### Pie Chart

```python
slices = [10, 20, 30, 5]
labels = ["A", "B", "C", "D"]
plt.pie(slices, labels=labels, autopct='%1.1f%%')
plt.title("Pie Chart")
plt.show()
```

### Scatter Plot

```python
colors = ['red' if y > df['y'].median() else 'blue' for y in df['y']]
plt.scatter(df['x'], df['y'], c=colors, s=120, edgecolor='k', alpha=0.8)
```

- `c`: colour array; `s`: marker size; `alpha`: transparency.

---

## Seaborn

Seaborn is designed for **tidy (long-form) data** and performs groupby automatically.

```python
import seaborn as sns

# Generic template
sns.someplot(x='col_x', y='col_y', data=df)

# Linear regression plot by group
sns.lmplot(x="total_bill", y="tip", col="day", hue="day", data=tips)
```

Seaborn integrates with Pandas DataFrames and handles the grouping logic internally. It produces publication-quality statistical plots with minimal code.

---

## Practical Exercise: Bird Flu Stacked Bar Chart

```python
# a. Load data
df = pd.read_csv('BirdFluCases.txt', sep='\t')

# b. Compute annual case counts
case_counts = df.groupby('Year')['Cases'].sum().reset_index()
death_counts = df.groupby('Year')['Deaths'].sum().reset_index()

# c. Merge
merged = pd.merge(case_counts, death_counts, on='Year')

years  = merged['Year'].values
cases  = merged['Cases'].values
deaths = merged['Deaths'].values

# d. Stacked bar
plt.bar(years, cases, label='Cases')
plt.bar(years, deaths, bottom=cases, label='Deaths')
plt.legend()
plt.title('Bird Flu Cases and Deaths by Year')
plt.show()

# e. Clustered bar
x = np.arange(len(years))
width = 0.35
plt.bar(x - width/2, cases, width, label='Cases')
plt.bar(x + width/2, deaths, width, label='Deaths')
plt.xticks(x, years)
plt.legend()
plt.show()
```

> The clustered version correctly conveys that deaths are a *subset* of cases. The stacked version visually inflates the apparent total and can mislead viewers into thinking cases + deaths are additive quantities.
