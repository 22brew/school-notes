---
title: "Assignment 1: Individual Project"
sidebar_label: "Assignment 1"
sidebar_position: 1
---

## Learning Objectives

- Use Python to perform data preparation
- Perform data exploration using charts and statistical measures
- Perform regression analysis
- Critically analyze data and come up with useful insights

## Overview

This assignment assesses your ability to apply what you have learned in the course (Python Programming, Data Preparation, and Regression). You will work with the **occupation dataset** (`occupations.csv`), which contains information on various occupations and their perceived prestige.

**Dataset:** `occupations.csv` — 98 records, each representing an occupational category.

| Variable | Description |
|---|---|
| occupation | The name or code of the occupation |
| education | Average education of occupational incumbents (years) |
| income | Average income of incumbents (dollars) |
| women | Percentage of incumbents who are women |
| perceivedprestige | Pineo-Porter prestige score from a mid-1960s social survey |
| census | Canadian Census occupational code |
| type | Occupation type: `bc` (Blue Collar), `pmt` (Professional/Managerial/Technical), `wc` (White Collar) |

## Questions

**2.** Using Python, load the dataset, prepare the data, perform data understanding, and data modeling.

### (a) Data Preparation

**i.** Identify all variables containing missing data and report their frequencies (counts).

**ii.** Implement an appropriate imputation strategy in Python. Provide clear justification for your chosen measures.

### (b) Data Exploration

Construct one visualization to analyze the distributional spread and median of perceived prestige across the three occupation type levels (`bc`, `wc`, `pmt`). The chart should clearly display the interquartile ranges and any outliers for each category. Interpret the chart clearly.

### (c) Handling Categorical Variables (Dummy Coding)

The variable `type` has three categories: Blue Collar, Professional/Managerial/Technical, and White Collar.

**i.** Use `'wc'` (white collar) as the reference level in subsequent sections.

### (d) Building a Model

Fit a linear regression model with `perceivedprestige` as the dependent variable and `education`, `log(income)`, and `type` as the independent variables. Save it as `model1`.

**i.** For which predictors can you reject the null hypothesis $H_0: \beta_j = 0$?

**ii.** Evaluate the overall model fit. Discuss whether this model provides a "meaningful" fit.

### (e) Interpretation

**i.** Provide and interpret the regression coefficient estimate for the dummy variable `typebc` in context.

**ii.** Provide and interpret the regression coefficient estimate for the dummy variable `typepmt` in context.

### (f) Statistical Significance

Assess the statistical significance of the `typebc` and `typepmt` regression coefficient estimates individually. Explain what their significance reveals. Support your answer with statistical evidence.

### (g) Building Another Model

Fit a new linear regression model to predict perceived prestige using `log(income)`, `type`, and an interaction term between occupation type and `log(income)`. Name this model `model2`.

**i.** Based on `model2`, what is the estimated effect of `log(income)` on perceived prestige for blue collar occupations?

**ii.** Write down the corresponding perceived prestige equation.

**iii.** How would you interpret the results?

---

## My Submission

<iframe
  src="/solutions/assignments/assignment1-submission.html"
  width="100%"
  height="900"
  frameborder="0"
  style="border: 1px solid #e0e0e0; border-radius: 4px;"
></iframe>
