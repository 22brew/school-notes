---
title: "Assignment 2: Individual Project"
sidebar_label: "Assignment 2"
sidebar_position: 2
---

## Learning Objectives

- Use Python to perform binary classification
- Interpret and evaluate model performance

## Overview

This assignment assesses your ability to apply what you have learned in the course (Python Programming and Logistic Regression). You will work with the **stroke dataset** (`stroke.csv`), which contains information on 213 hospitalized stroke patients.

**Dataset:** `stroke.csv` — 213 records, each representing a patient.

| Variable | Description |
|---|---|
| status | Patient status during hospitalization (`alive` or `dead`) |
| sex | Sex of the patient (`male` or `female`) |
| dm | History of diabetes mellitus (`yes` or `no`) |
| gcs | Glasgow Coma Scale score (3–15; lower = more severe impairment) |
| sbp | Systolic blood pressure on admission (mmHg) |
| dbp | Diastolic blood pressure on admission (mmHg) |
| wbc | White blood cell count on admission |
| time2 | Time elapsed (days) between symptom onset and hospitalization |
| stroke_type | Type of stroke: `IS` (Ischaemic) or `HS` (Haemorrhagic) |
| referral_from | Source of referral or admission |

> **Rounding:** If the answer is > 1, round to 2 decimal places. If < 1, round to 3 significant figures.

## Questions

### (a) Dependent Variable Preparation

Create a new binary variable, `mortality_status`, where `1 = dead` and `0 = alive`. Add it to the dataframe. Print the unique values and a frequency table to verify the mapping.

### (b) Categorical Encoding and Logistic Regression Modeling

Build a model to predict the probability of mortality using clinical predictors.

- **Model Specification:** Using `statsmodels.formula.api`, fit a multiple logistic regression (Binomial GLM) to predict `mortality_status`.
- **Predictors:** Include `stroke_type`, `gcs`, `sbp`, `dbp`, and `wbc`.
- **Handling Categorical Data:** Treat `stroke_type` as categorical. Explicitly set `"IS"` (Ischaemic Stroke) as the reference (baseline) category.

Fit the model `model1`, write down the estimated logistic regression equation in logit form, and display the model summary output.

### (c) Model Interpretation

Based on your fitted logistic regression model in (b):

- **GCS:** Report and interpret the logistic regression coefficient estimate for `gcs` in terms of odds of mortality.
- **Statistical Significance:** Determine if GCS is a statistically significant predictor of mortality.
- **Stroke Type:** Report and interpret the coefficient estimate for `stroke_type = Haemorrhagic Stroke [T.HS]` in terms of odds of mortality and its clinical implication.

### (d) Evaluating Interaction Effects

Extend your model by including an interaction term between `stroke_type` and `gcs`. Decide whether to retain the interaction term in your final model and provide a justification based on your results.

### (e) Classification and Confusion Matrix

Using `model1`, generate the predicted probability of mortality for each patient.

- Apply a classification threshold: classify as `1 (Death)` if predicted probability ≥ 0.50, `0 (Alive)` if < 0.50.
- Build a confusion matrix comparing actual `mortality_status` with `predicted_mortality_status`.

> **Note:** Ensure `1 (Death)` is treated as the Positive Class.

Clearly state the values for True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

Using the confusion matrix, compute and report:
- **Accuracy**
- **Recall (Sensitivity)**
- **Specificity**

Explain what each metric represents specifically in the context of predicting stroke mortality. Discuss any trade-offs between Sensitivity and Specificity. In a medical emergency setting, which metric would you prioritize? Provide a justification.

---

## My Submission

<iframe
  src="/solutions/assignments/assignment2-submission.html"
  width="100%"
  height="900"
  frameborder="0"
  style="border: 1px solid #e0e0e0; border-radius: 4px;"
></iframe>

## Official Solution

<iframe
  src="/solutions/assignments/assignment2-solution.html"
  width="100%"
  height="900"
  frameborder="0"
  style="border: 1px solid #e0e0e0; border-radius: 4px;"
></iframe>
