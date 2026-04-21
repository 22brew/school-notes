---
title: "L2: Python Basics I"
sidebar_label: "L2: Python Basics I"
sidebar_position: 2
---

## Why Python for Analytics?

Most analytics workflows require programming to:
- Filter, merge, and clean data.
- Generate and compare predictive models.
- Reproduce experiments systematically.

This course uses **Python** — an open-source language with a rich ecosystem for data science. Python 2.0 was released in 2000; the current standard is Python 3. Its community-driven development model means new statistical and ML libraries emerge constantly.

**Key applications in analytics:**
- Statistical analysis
- Data visualisation
- Machine learning and predictive modelling

---

## Setting Up: Anaconda

Use the **Anaconda distribution** — it bundles 250+ data science packages and simplifies environment management.

- **Conda:** CLI environment manager.
- **Anaconda Navigator:** GUI for managing environments and IDEs (Jupyter, JupyterLab).
- **Jupyter Notebook:** Ideal for interactive, reproducible data science.

```python
# Install packages in Jupyter using pip
!pip install pandas seaborn statsmodels scikit-learn
```

Download: [anaconda.com/download](https://www.anaconda.com/download)

> **Tip:** Use `!pip install` inside a Jupyter cell (the `!` runs it as a shell command).

---

## Basic Python Data Structures

Python has several built-in types:

| Type | Description | Example |
|------|-------------|---------|
| `int` | Integer | `42` |
| `float` | Floating point | `3.14` |
| `bool` | Boolean | `True`, `False` |
| `str` | String | `"hello"` |
| `list` | Ordered, mutable | `[1, 2, 3]` |
| `tuple` | Ordered, immutable | `(1, 2, 3)` |
| `dict` | Key-value mapping | `{"a": 1, "b": 2}` |

**Mutability:** Lists are mutable (you can change elements); tuples are not. **Aliasing** means two variables can point to the same underlying object — be careful when passing lists to functions.

---

## NumPy

**NumPy** (Numerical Python) is the foundation of scientific computing in Python. It provides:
- High-performance multidimensional array objects (`ndarray`).
- Fast operations on large data arrays **without** explicit loops.
- **Vectorisation:** operations run on pre-compiled C code — orders of magnitude faster than pure Python loops.
- **Broadcasting:** smart element-wise operations across arrays of different shapes.

```python
import numpy as np
```

### Array Concepts

| Concept | Description |
|---------|-------------|
| **Rank** | Number of dimensions (axes) |
| **Shape** | Tuple describing size in each dimension |
| **Size** | Total number of elements |

```python
# 1D array — shape (5,), size 5
arr1d = np.array([1, 2, 3, 4, 5])

# 2D array — shape (2, 4), size 8
arr2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Utility constructors
np.ones((2, 2), dtype=int)     # [[1,1],[1,1]]
np.zeros((3, 3))               # 3x3 zeros
np.arange(1, 10, 2)            # [1, 3, 5, 7, 9]
```

### Reshaping

```python
arr = np.arange(1, 7)
arr.reshape(2, 3)
# array([[1, 2, 3],
#        [4, 5, 6]])
```

### Axes and Reductions

```python
a = np.array([[1, 2], [3, 4]])
a.sum()           # 10  — sum all
a.sum(axis=0)     # [4, 6]  — sum over rows (collapse rows)
a.sum(axis=1)     # [3, 7]  — sum over columns (collapse columns)
a.sum(axis=1, keepdims=True)  # [[3], [7]]
```

### Vectorisation vs Loops

```python
a = np.random.random(500_000)
b = np.random.random(500_000)

# Vectorised: ~2.9 ms
result = np.dot(a, b)

# Loop: ~345 ms
dot = 0.0
for i in range(len(a)):
    dot += a[i] * b[i]
```

Vectorisation is **~100× faster** for large arrays.

### Vectorised Operations

```python
x = np.array([[1, 2], [3, 4]])
y = np.ones((2, 2))
np.subtract(x, y)
# array([[0., 1.],
#        [2., 3.]])
```

### Broadcasting Rules

Broadcasting allows operations between arrays of different shapes:

1. If dimensions are equal → no change.
2. If one dimension is 1 → it gets repeated (broadcast) to match.
3. If dimensions are unequal and neither is 1 → raises an error.

```python
a1 = np.arange(1, 10).reshape(3, 3)  # shape (3, 3)
a2 = np.array([10, 10, 10])           # shape (3,)
a1 + a2
# array([[11, 12, 13],
#        [14, 15, 16],
#        [17, 18, 19]])
```

`a2` is broadcast row-by-row across `a1`.

### Concatenation

```python
A = np.array([[1, 2], [3, 4]])  # shape (2, 2)
B = np.array([[5, 6]])          # shape (1, 2)

# axis=0 → join vertically
C = np.concatenate((A, B), axis=0)
# shape (3, 2): [[1,2],[3,4],[5,6]]

# axis=1 → join horizontally (requires same number of rows)
```

---

## Pandas

**Pandas** is the go-to Python library for data manipulation and analysis. Built on NumPy, it provides:
- Efficient data structures: `Series` (1D) and `DataFrame` (2D).
- Handling of missing data (`.isnull()`, `.dropna()`, `.fillna()`).
- Flexible grouping and aggregation (`.groupby()`, `.agg()`).
- Efficient merging (`.merge()`, `.join()`).

```python
import pandas as pd
```

### Core Data Structures

**Series** — 1D array with an index:
```python
a = [7, 4, 2, 22, -12]
x = pd.Series(a)
# 0     7
# 1     4
# 2     2
# 3    22
# 4   -12
x[1]    # 4
```

**DataFrame** — 2D table of named Series sharing the same index:
```python
type(df)             # pandas.core.frame.DataFrame
type(df["Column"])   # pandas.core.series.Series
```

### Reading Data

```python
import os
print(os.getcwd())       # Check working directory
os.chdir('path/to/dir') # Change directory (use forward slashes)

df = pd.read_csv('elections.csv', header=0)
df.shape       # (rows, columns)
df.describe()  # Summary statistics
df.head()      # First 5 rows
df.tail()      # Last 5 rows
```

### Indexing: `[ ]` Operator

```python
df["Candidate"]              # → Series
df[["Candidate", "Party"]]   # → DataFrame
df[0:3]                      # → First 3 rows (slice)
```

### Label-Based Indexing: `.loc`

`.loc` selects by **label** (index name or column name):

```python
df.loc[[0, 1, 2], ['Candidate', 'Party', 'Year']]  # rows 0–2, 3 columns
df.loc[0:4, 'Candidate']      # → Series (single column label)
df.loc[0:4, ['Candidate']]    # → DataFrame (list of one column)
df.loc[[0], 'Candidate':'Year']  # → DataFrame, shape (1, 4)
```

**Boolean indexing with `.loc`:**
```python
df.loc[(df['Result'] == 'win') & (df['%'] < 50), 'Candidate':'%']
```

### Position-Based Indexing: `.iloc`

`.iloc` selects by **integer position**:

```python
df.iloc[0:3, 0:3]   # First 3 rows, first 3 columns
```

| Method | Selects by | Note |
|--------|-----------|------|
| `.loc` | Label | Inclusive of end label |
| `.iloc` | Integer position | Exclusive of end index |

Prefer `.loc` — harder to make mistakes, and not vulnerable to reordering of columns in the raw file.

### Setting the Index

```python
elections.set_index("Candidate")  # Set Candidate column as row labels
```

### Data Aggregation: `groupby`

```python
grouped = df.groupby('Party')

# Iterate over groups
for name, group in grouped:
    print(name)
    print(group.head())

# Aggregate
df.groupby('Party')[['%', 'Year']].mean()
```

`groupby` splits the DataFrame into groups, you apply an aggregation function (`.mean()`, `.sum()`, `.count()`, etc.), and Pandas combines the results back into a DataFrame.

---

## Practice Questions

**Q1.** Suppose you have two NumPy arrays:
```python
A = np.array([[1, 2], [3, 4]])   # shape (2, 2)
B = np.array([[5, 6]])            # shape (1, 2)
C = np.concatenate((A, B), axis=0)
```
What is the shape and content of `C`?

- A) A `ValueError` is raised.
- **B) ✓** Shape `(3, 2)`. `C = [[1, 2], [3, 4], [5, 6]]`
- C) Shape `(2, 3)`. `C = [[1, 2, 5], [3, 4, 6]]`
- D) Shape `(3, 2)`. `C = [[5, 6], [1, 2], [3, 4]]`

> **Why B:** `axis=0` stacks vertically. Both arrays have 2 columns, so they're compatible. B is appended below A.

---

**Q2.** Consider `df.loc[[0], 'Candidate':'Year']`. If `'Candidate'` is the 1st column and `'Year'` is the 4th column, what is the shape of the returned object?

- A) `(4,)` — a Series with 4 elements.
- **B) ✓** `(1, 4)` — a DataFrame with 1 row and 4 columns.
- C) `(0, 4)` — an empty DataFrame because index 0 is excluded.
- D) I'm not sure.

> **Why B:** Providing a list `[0]` as the row selector returns a DataFrame (not a Series). `.loc` with a label slice `'Candidate':'Year'` is **inclusive** of both endpoints, giving 4 columns.
