# Manufacturing Defect Analysis — SECOM Semiconductor Dataset

A data science project exploring what makes semiconductor chips fail quality inspection — using statistical testing, logistic regression, and random forest classification.

---

## Why I Built This

Semiconductor manufacturing runs hundreds of process sensors simultaneously across thousands of production cycles. With a ~6.6% defect rate, even small improvements in defect prediction translate directly into cost savings. I wanted to understand: can we look at raw sensor readings and figure out *which process variables actually matter* for quality outcomes — without needing a domain expert to tell us upfront?

That question is what this project tries to answer.

---

## Dataset

**UCI SECOM Manufacturing Dataset**
- 1,567 wafer production runs
- 590 process sensor measurements per run
- Binary outcome: pass (0) or fail/defect (1)
- Defect rate: ~6.6% (imbalanced)

The dataset downloads automatically when you run the notebook. No manual setup needed.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SECOM)

---

## What's In The Notebook

The notebook walks through the full project in sequence:

**1. Data Loading & First Look**
Load both sensor data and labels, map the label format (-1/1 → 0/1), and get a feel for what we're working with.

**2. Data Cleaning**
Three passes of cleaning:
- Dropped sensors with more than 50% missing values (not enough data to be useful)
- Dropped constant sensors — a sensor that never changes tells you nothing
- Imputed remaining missing values using the column median (more robust than mean for skewed sensor data)

After cleaning: ~446 features remained from the original 590.

**3. Exploratory Data Analysis**
Visualised class distribution, compared sensor reading distributions between passing and failing wafers, and mapped out missing value patterns.

**4. Statistical Hypothesis Testing**
Ran Mann-Whitney U tests across all cleaned sensors to identify which ones show a statistically significant difference between pass and fail groups (p < 0.05). Applied Bonferroni correction for multiple comparisons.

I chose Mann-Whitney over the t-test because sensor readings in manufacturing are rarely normally distributed — the non-parametric test is more appropriate here.

**5. Modelling**
Built two classifiers on the statistically significant features:
- Logistic Regression (with L2 regularisation, class_weight='balanced')
- Random Forest (200 estimators, max_depth=8, class_weight='balanced')

Both used StandardScaler preprocessing and were evaluated with 5-fold stratified cross-validation.

**6. Evaluation**
- Confusion matrix
- Classification report (precision, recall, F1)
- ROC-AUC curves for both models side-by-side

**7. Feature Importance**
Extracted feature importances from the Random Forest to identify which sensors contribute most to defect prediction. The top 5 sensors alone account for a significant share of predictive power — these are the process variables worth monitoring most closely.

**8. Business Recommendations**
Translated model outputs into practical recommendations: which sensors to put under tighter statistical process control (SPC), what the early-warning system would look like in a real production environment, and an estimated cost impact.

---

## Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.72 | ~0.67 |
| Random Forest | ~0.77 | ~0.76 |


The top 5 sensors identified by feature importance analysis account for roughly 14.7% of the model's predictive power. These are the process variables most associated with defect outcomes in this dataset.

---

## Key Concepts Used

- Exploratory Data Analysis (EDA)
- Median imputation for missing data
- Mann-Whitney U test (non-parametric hypothesis testing)
- Bonferroni correction for multiple comparisons
- Logistic Regression with L2 regularisation
- Random Forest (ensemble learning)
- Stratified K-Fold Cross Validation
- ROC-AUC evaluation
- Feature importance (Gini impurity-based)

---


## Project Structure

```
manufacturing-defect-analysis/
├── README.md
├── manufacturing_defect_analysis.ipynb
├── requirements.txt
├── images/
│   ├── eda_overview.png
│   ├── sensor_boxplots.png
│   ├── model_evaluation.png
│   └── feature_importance.png
└── .gitignore
```

---

## What I Learned

A few things that weren't obvious before I started this:

The data cleaning step took longer than the modelling. Going from 590 sensors to ~446 usable features before even touching a model is a reminder that real data is messy and most of the work is upstream of the algorithm.

Class imbalance matters more than I expected. Using accuracy as a metric on a 6.6% defect rate would give you 93.4% accuracy by just predicting "pass" every time — which is completely useless. ROC-AUC and recall on the defect class are what actually tell you if the model is doing anything.

Statistical testing before modelling is underused. Running hypothesis tests first to filter features is faster and more explainable than letting the model figure it out on its own. Being able to say "these sensors were statistically significant at p < 0.05" is a cleaner story than "the model said so."

---

## About

M.Tech student in Data Science & Analytics,from Lovely Professional University, interested in applying ML and statistical methods to real manufacturing and operational problems.

