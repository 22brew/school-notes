---
title: "L1: Introduction"
sidebar_label: "L1: Introduction"
sidebar_position: 1
---

## Introduction to Analytics

**Analytics** is the use of data, information technology, statistical analysis, and mathematical or computer-based models to support decision making.

Companies generate enormous volumes of data — sales, transactions, customer records — and data about companies is also produced externally (e.g., social media, reviews, market data). The challenge is extracting actionable insight from this raw information.

**Decision Making** is the process of choosing among two or more possible actions to attain a goal. Analytics provides a scientific, systematic foundation for those decisions, drawing on disciplines including statistics, economics, computer science, operations research, and psychology.

---

## Types of Decision Problems

### Supervised Learning

In supervised learning, we have a labelled dataset of observations $\{(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)\}$ and use it to build a model that predicts $Y$ for new, unseen data.

- **Regression** — response $Y$ is quantitative (continuous).  
  *Example:* Predicting home prices from square footage, number of bedrooms, location.

- **Classification** — response $Y$ takes values in a finite, unordered set.  
  *Example:* Predicting whether a patient has heart disease (Positive/Negative) from obesity index, age, family history.

### Unsupervised Learning

In unsupervised learning, we observe only the features $X$ with no measured outcome $Y$. The goal is to discover structure — how the data are organised or clustered — rather than predict a labelled response.

- **Clustering** — group observations sharing common characteristics.  
  *Example:* Grouping Facebook users by interests; segmenting customers by spending patterns.

Because there is no ground truth $Y$ to measure against, unsupervised methods are inherently harder to evaluate — this is what ISLP means by "working blind."

### Summary of Problem Types

| Type | Output | Example |
|------|--------|---------|
| Regression (Supervised) | Continuous value | Price prediction |
| Classification (Supervised) | Discrete category | Weather: sunny/rainy/cloudy |
| Clustering (Unsupervised) | Group labels | Customer segmentation |

### Training vs Testing Data

- **Training data:** Labelled data used to fit (learn) the model.
- **Testing data:** Unseen data used to evaluate how well the fitted model generalises.

> **Key insight:** A model's in-sample (training) performance is not a reliable guide to out-of-sample (test) performance. This is why test sets, cross-validation, and resampling methods matter.

---

## Statistical Learning

### The Core Framework

Statistical learning assumes a systematic relationship between a quantitative response $Y$ and predictors $X = (X_1, \ldots, X_p)$:

$$Y_i = f(X_i) + \varepsilon_i$$

where:
- $f$ is the unknown function capturing the systematic dependence of $Y$ on $X$.
- $\varepsilon$ is a random error term with mean zero, representing unobservable variation.

Our job is to **estimate $f$** from training data.

### Why Estimate $f$?

There are two primary reasons:

**1. Prediction**

When we don't care about the form of $f$ — only its output — we treat $\hat{f}$ as a black box:

$$\hat{Y}_i = \hat{f}(X_i)$$

Prediction accuracy depends on two error sources:
- **Reducible error:** Comes from $\hat{f}$ not perfectly estimating $f$. Can be improved with better models/data.
- **Irreducible error:** Comes from $\varepsilon$ itself. No model can eliminate this — it is noise inherent in the data.

**2. Inference**

When we want to understand *how* $Y$ relates to $X_1, \ldots, X_p$:
- Which predictors actually affect $Y$?
- Is the relationship positive or negative?
- Is a linear model adequate?

Interpretable models (e.g., linear regression, logistic regression) support inference; complex black-box models (e.g., gradient boosting, neural networks) prioritise prediction accuracy.

### Bias–Variance Tradeoff

A key concept underlying model selection: more flexible models have lower bias (they fit the training data closely) but higher variance (they change a lot across different training sets, i.e., they overfit). Simpler models have higher bias but lower variance. The goal is to find the optimal balance that minimises total test error. This tradeoff is fundamental to virtually every modelling decision in this course.

---

## Applications of Analytics

### Customer Relationship Management
- Maximise return on marketing campaigns.
- Improve customer retention (churn analysis).
- Maximise customer value through cross-selling and up-selling.
- Identify and treat the most valuable customers.

### Banking and Financial Services
- Automate loan application processing.
- Detect fraudulent transactions.
- Optimise cash reserves through forecasting.

### Retailing and Logistics
- Optimise inventory levels across locations.
- Improve store layout and promotional effectiveness.
- Minimise losses from perishable goods.
- Optimise logistics by predicting seasonal demand.

### Manufacturing and Maintenance
- Predict and prevent machinery failures (predictive maintenance).
- Identify anomalies in production systems.
- Discover patterns that improve product quality.

### Brokerage and Securities
- Forecast bond price changes.
- Assess the effect of events on market movements.
- Identify and prevent fraudulent trading.

### Insurance
- Forecast claim costs for business planning.
- Determine optimal rate plans.
- Identify and prevent fraudulent claims.

---

## Data Mining Process: CRISP-DM

The **Cross-Industry Standard Process for Data Mining (CRISP-DM)** is the most widely adopted framework for data science projects. It breaks the process into six iterative phases — "iterative" because in practice you will often step back to earlier phases as you learn more.

### Phase 1: Business Understanding
Understand the *purpose* of the study before touching data.
- Define project objectives and business requirements.
- Form a rough idea of what data might be relevant.
- Produce a preliminary plan.

> **Key takeaway:** Start with the problem, not the data. Solving the wrong problem is worse than not solving at all.

### Phase 2: Data Understanding
Identify, collect, and explore the relevant data.
- Describe attributes, data types, and ranges.
- Explore relationships using graphs (scatter plots, histograms) and statistics (correlations, summary stats).
- Identify potential quality issues early.

### Phase 3: Data Preparation
Transform raw data into a form suitable for modelling. This is typically the **most time-consuming** phase and has the largest impact on model accuracy.

- **Data consolidation:** Combine data from multiple sources.
- **Data cleaning:** Handle missing values, noisy data, and inconsistencies.
- **Data transformation:** Normalise, discretise, aggregate, or engineer new features.
- **Data reduction:** Remove irrelevant features or records; balance class distributions.

### Phase 4: Model Building
Select and apply statistical learning techniques. Some algorithms have specific data format requirements, so iteration back to Data Preparation is common.

### Phase 5: Testing and Evaluation
Assess model performance using appropriate metrics:
- **Regression:** Mean Squared Error (MSE), R², adjusted R².
- **Classification:** Accuracy, precision, recall, F1, AUC-ROC.

Evaluation uses held-out test data or cross-validation — never the same data used to train.

### Phase 6: Deployment
Put the model into production. Complexity varies:
- Simple: Generate a report.
- Complex: Integrate into a real-time operational system.

Ongoing **monitoring and maintenance** is essential — models decay as the real world changes.

### CRISP-DM Key Takeaways
1. The process starts with **understanding the problem**, not the data.
2. Understand the data thoroughly before modelling.
3. The process does **not** end at model building — always evaluate and monitor.

---

## Practice Questions

**Q1.** The text *Introduction to Statistical Learning* states that "in unsupervised learning, we are in some sense working blind." Which fundamental challenge does this description most directly refer to?

- A) The computational complexity of unsupervised algorithms is significantly higher.
- **B) ✓** The absence of a response variable $Y$ eliminates the possibility of defining a unique loss function for model validation and selection, forcing reliance on heuristic or internal criteria.
- C) Unsupervised methods cannot be used for inference, only prediction.
- D) The results of an unsupervised analysis are uninterpretable.

> **Why B:** Without a labelled $Y$, there is no external benchmark to measure whether the discovered structure (e.g., clusters) is "correct." Evaluation must rely on internal metrics (e.g., within-cluster variance) or domain judgement, making the problem fundamentally harder to validate.

---

**Q2.** A research team is studying the impact of long-term PM2.5 exposure on cardiovascular risk. They want to test whether PM2.5 is *significantly associated* with cardiovascular risk. Which modelling strategy best supports this goal?

- A) Use gradient boosting to confirm PM2.5's significance.
- **B) ✓** Use a simple, interpretable model (e.g., logistic regression) to quantify PM2.5's effect with confidence intervals and p-values.
- C) Use a neural network to understand PM2.5's impact.
- D) I'm not sure.

> **Why B:** Inference requires interpretable models that produce coefficient estimates with well-defined standard errors and p-values. Complex models like neural networks do not provide straightforward statistical tests of individual predictors.

---

**Q3.** A team wants to develop a clinical tool that accurately predicts an individual's 10-year risk of a heart attack. Which modelling strategy best supports this *prediction* objective?

- A) Irreducible error makes accurate prediction fundamentally impossible.
- B) Use a simple linear model for trustworthiness, despite lower performance.
- **C) ✓** Use a highly complex, flexible model (e.g., Gradient Boosting) to maximise predictive accuracy.
- D) I'm not sure.

> **Why C:** When the primary goal is prediction accuracy (not inference), flexible models that capture complex non-linear patterns are preferred, provided they are properly validated on held-out test data to avoid overfitting.
