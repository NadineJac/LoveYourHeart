import streamlit as st
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Page config
st.set_page_config(page_title="Documentation", page_icon="🫀", layout="wide")
st.markdown("""
# 🫀 Love Your Heart - Heart Disease Risk Prediction Using Machine Learning

A comprehensive end-to-end health analytics & ML project focused on predicting a person's heart attack riskusing clinical, lifestyle, and demographic def-reprot data.

## 📌 Table of Contents
- [🫀 Love Your Heart - Heart Disease Risk Prediction Using Machine Learning](#-love-your-heart---heart-disease-risk-prediction-using-machine-learning)
  - [📌 Table of Contents](#-table-of-contents)
  - [🧩 Project Overview](#-project-overview)
  - [❗ Problem Statement](#-problem-statement)
  - [📊 Dataset Description](#-dataset-description)
  - [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
    - [1️⃣ Class Imbalance](#1️⃣-class-imbalance)
    - [2️⃣ Numerical Feature Distributions](#2️⃣-numerical-feature-distributions)
    - [3️⃣ Categorical Feature Distributions](#3️⃣-categorical-feature-distributions)
    - [4️⃣ 📌 Correlation Heatmap](#4️⃣--correlation-heatmap)
  - [🧼 Data Preprocessing](#-data-preprocessing)
  - [🤖 Modeling Approach](#-modeling-approach)
    - [🎯 Hyperparameter Tuning](#-hyperparameter-tuning)
      - [1. Grid Search \& Randomized Search](#1-grid-search--randomized-search)
        - [2. Class Weights](#2-class-weights)
      - [3. scale\_pos\_weight in XGBoost](#3-scale_pos_weight-in-xgboost)
      - [4. Iteration \& Depth Tuning for CatBoost](#4-iteration--depth-tuning-for-catboost)
      - [5. Propability Calibration](#5-propability-calibration)
    - [✅ Overall Tuning Outcome](#-overall-tuning-outcome)
    - [📈 Model Comparison](#-model-comparison)
    - [🎯 Threshold Optimization](#-threshold-optimization)
  - [🧠 Explainability Using SHAP](#-explainability-using-shap)
  - [🤖 CHATBOT Module:](#-chatbot-module)
    - [📈 Workflow Diagrams:](#-workflow-diagrams)
      - [1️⃣ System Architecture Diagram](#1️⃣-system-architecture-diagram)
      - [2️⃣ Chatbot Workflow Diagram](#2️⃣-chatbot-workflow-diagram)
  - [🚀 Final Model Deployment](#-final-model-deployment)
  - [Limitations](#limitations)
  - [🔮 Future Work](#-future-work)
  - [🏁 Conclusion](#-conclusion)

## 🧩 Project Overview

We build a machine learning model that predicts the likelihood of heart disease based on a users lifestyle, medical history, and health metrics.
The model is deployed as an interactive Streamlit-app allow users to assess their risk score.
This documentation summarizes the entire workflow, decisions taken, trade-offs, and final model deployment strategy.

## ❗ Problem Statement

Heart disease, including heart attacks, remains one of the world’s leading causes of death. Early detection can significantly improve survival rates. The goal of this project is to:
* Build a predictive model that produces a probability score of belonging to the heart attack class (0–100%)
* Minimize false negatives (missed high-risk cases)
* Provide users with explainable and actionable insights

## 📊 Dataset Description

**Dataset used:** BRFSS_2020_Heart_Disease_Dataset - https://zenodo.org/records/15364962
Contains 320,000+ rows and 18 features across:

**Demographic factors:** sex, age category (14 levels), race, BMI (Body Mass Index)

**Diseases:** weather respondent ever had such diseases as **asthma, skin cancer, diabetes, stroke or kidney disease** (not including kidney stones, bladder infection or incontinence)

**Unhealthy habits:**
* **Smoking** - respondents that smoked at least 100 cigarettes in their entire life (5 packs = 100 cigarettes)
* **Alcohol Drinking** - heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week).
  
**General Health:**
* **Difficulty Walking** - weather respondent have serious difficulty walking or climbing stairs
* **Physical Activity** - adults who reported doing physical activity or exercise during the past 30 days other than their regular job
* **Sleep Time** - respondent’s reported average hours of sleep in a 24-hour period
* **Physical Health** - number of days being physically ill or injured (0-30 days)
* **Mental Health** - number of days having bad mental health (0-30 days)
* **General Health** - respondents declared their health as ’Excellent’, ’Very good’, ’Good’ ,’Fair’ or ’Poor’

## 🔍 Exploratory Data Analysis (EDA)
The dataset consists of both categorical and numerical features representing demographic, lifestyle, and health-related attributes. 

### 1️⃣ Class Imbalance

Heart disease is heavily imbalanced (only ~8% "Yes").
""")
IMG_PATH = os.path.join(PROJECT_ROOT, "images", "img_class_distribution.png")
st.image(IMG_PATH, caption="distribution of target lable heart attack")

st.markdown("""

![distribution of target lable heart attack](images/img_class_distribution.png)

Only 8% of the dataset represents heart-attack cases, creating a strong class imbalance. To handle this, techniques like SMOTE (Synthetic Minority Over-sampling Technique), class-weighted models, and threshold tuning were applied to improve detection of high-risk individuals while maintaining overall accuracy. Properly addressing this imbalance is crucial to avoid missing critical high-risk cases.

### 2️⃣ Numerical Feature Distributions
Numerical features, including BMI, Age, and PhysicalHealth, display distinct distributions that help identify risk patterns. 

![Age Distribution](../images/img_age_distribution.png)

![BMI distribution](images/img_bmi_distribution.png)

### 3️⃣ Categorical Feature Distributions
Categorical features such as Sex, Smoking, AlcoholDrinking, and Diabetic exhibit varying proportions across different classes, highlighting trends like higher smoking prevalence among certain groups. 

![Categorical Feature Distributions](images/img_catFeature_distribution.png)

### 4️⃣ 📌 Correlation Heatmap

![Correlation Heatmap](images/img_corr_heatmap.png)

The heatmap helps identify which features are most strongly related to heart disease risk. Features like age, sex, smoking, alcohol comsumption, are positively correlated with risk, while lifestyle-related features such as physical activity and adequate sleep are negatively correlated. These insights can guide feature selection
and interpretation in predictive modeling.

## 🧼 Data Preprocessing

1. Train-test split (80–20, stratified)
2. Preprocessor
   1. Handling categorical variables via OneHotEncoder
   2. Scaling numerical values
3. SMOTE (optional depending on model)
4. Voting classifier

![Pipeline](images/img_pipeline.png)

## 🤖 Modeling Approach

We used 5 major models:

| Model               | Notes                                                   |
| ------------------- | ------------------------------------------------------- |
| Logistic Regression | Baseline, interpretable                                 |
| Random Forest       | Good recall, handles non-linearities                    |
| Gradient Boosting   | Stable performance, robust to overfitting               |
| XGBoost             | Strong tree-based model, fast and scalable              |
| CatBoost            | Excellent with categorical data, handles missing values |

### 🎯 Hyperparameter Tuning

To improve model performance and ensure robust generalization, several hyperparameter optimization techniques were applied. The tuning process focused on improving recall for the minority class while maintaining acceptable precision.

#### 1. Grid Search & Randomized Search
Both *GridSearchCV* and *RandomizedSearchCV* were used depending on the model complexity:
*Grid Search* was applied to simpler models like Logistic Regression and Random Forest, where the parameter space was small.
*Randomized Search* was used for Gradient Boosting and XGBoost due to their larger parameter space, allowing faster exploration of combinations.

These search strategies helped identify optimal parameter ranges such as learning rate, tree depth, and number of estimators.

##### 2. Class Weights
Because the dataset is highly imbalanced (only ~8% positive cases), class weights were introduced to make the models more sensitive to minority-class predictions:
Models like Logistic Regression, Random Forest, and GradientBoosting were trained with class_weight='balanced'.

This ensures higher penalties for misclassifying positive (risk) cases, improving recall.

#### 3. scale_pos_weight in XGBoost
For XGBoost, the imbalance was handled using scale_pos_weight, computed as:
```
scale_pos_weight = number of negative samples / number of positive samples
```

This parameter tells XGBoost to give more importance to the minority class, improving risk detection without excessively increasing false positives.

#### 4. Iteration & Depth Tuning for CatBoost
CatBoost, which handles categorical data efficiently, was tuned for:
* **iterations** (number of trees) – higher iterations improved performance but were controlled to avoid overfitting.
* **depth** – deeper trees improved recall but increased variance, so an optimal mid-range depth was selected.
  
Additional parameters like learning rate and class weights were adjusted to maximize recall on the minority class.

#### 5. Propability Calibration
Tree-based models (RF/GB/XGB) tend to output uncalibrated, overconfident probabilities, so we applied: *CalibratedClassifierCV*

Two calibration methods were used:

- Isotonic Regression (more flexible, better for medium-large datasets)
- Sigmoid (Platt Scaling) (more stable for smaller datasets)

Purpose of Calibration

- Ensures a model score of 0.70 truly reflects ~70% risk
- Improves fairness across categories
- Makes thresholding (0.2, 0.35, 0.5) more reliable
- Builds user trust in the risk score displayed in the app

### ✅ Overall Tuning Outcome

By combining these techniques, each model was optimized to strike a balance between:
* High recall (detect as many true high-risk cases as possible),
* Controlled false positives, and
* Stable generalization performance.
  
This tuning strategy ultimately enabled the Voting Classifier to outperform individual models in terms of balanced performance.

### 📈 Model Comparison

![Model Comparison](images/img_model_comparison.png)

After evaluating all models across multiple performance metrics, we observed that XGBoost and CatBoost achieved the highest recall scores, making them strong candidates for identifying high-risk individuals. However, both models also produced a larger number of false positives, which can reduce the overall precision of the system.

To balance this trade-off, we incorporated multiple strong learners into a Voting Classifier, combining the strengths of Logistic Regression, Random Forest, XGBoost, and CatBoost. The ensemble model demonstrated more stable performance, achieving a significantly better balance between recall and false-positive control.

Because of this improved overall reliability and its ability to maintain high recall while reducing false alarms, the Voting Classifier was selected as the final model for deployment.

### 🎯 Threshold Optimization

![Precision-Recall Curve](images/img_precision_recall.png)

In highly imbalanced classification tasks such as heart-disease risk prediction, relying on the default probability threshold of 0.50 often results in a large number of false negatives — cases where high-risk individuals are incorrectly predicted as low-risk. Since missing a positive (high-risk) case has much higher clinical impact, threshold optimization becomes essential.

To address this, we evaluated the model using the Precision–Recall (PR) Curve, which is more informative than ROC-AUC for imbalanced datasets. The Voting Classifier achieved an Average Precision (AP) of 0.194, indicating moderate ability to distinguish minority-class samples across varying thresholds.

By analyzing the PR curve, we identified that lowering the threshold significantly improved recall without causing an unmanageable drop in precision. The optimal balance was achieved at a threshold of 0.19, where:
- Recall increased substantially, capturing far more true high-risk cases

- Precision remained acceptable, keeping false positives at a manageable level

- False negatives were reduced, which is critical for a health-risk prediction system

Therefore, instead of using the standard 0.50 cutoff, the model operates at a custom-calibrated threshold of 0.19, ensuring better sensitivity toward the minority (positive) class while maintaining practical precision levels. This threshold strategy aligns with our project’s objective: prioritizing early detection of high-risk individuals.

At this threshold:
* Higher recall
* Acceptable precision
* Balanced trade-off

## 🧠 Explainability Using SHAP


-- TO BE WRITTEN --



## 🤖 CHATBOT Module:

Our application also includes an AI-powered Chatbot designed to provide personalized recommendations based on user-specific health inputs. The chatbot is implemented using a Retrieval-Augmented Generation (RAG) pipeline, ensuring that responses are accurate, context-aware, and grounded in the domain knowledge relevant to heart-risk assessment.

To enhance its reasoning capabilities, the chatbot leverages modern Large Language Models (LLMs) and integrates multiple API sources, including the GROQ API and HUGGINGFACE models, enabling a balanced blend of speed, accuracy, and medical-focused contextual understanding. User interactions and prior results are stored through a controlled memory component, allowing the system to maintain continuity and deliver more tailored insights without compromising privacy.

Overall, the chatbot acts as an interactive layer of support—helping users interpret their predictions, understand lifestyle implications, and receive personalized guidance throughout the risk-screening process.

### 📈 Workflow Diagrams:

#### 1️⃣ System Architecture Diagram
```
+-----------------+
|   User Interface| <-- Streamlit (TestYourself / AI Assistant pages)
+-----------------+
          |
          v
+--------------------------+
|  Preprocessing Pipeline  | <-- StandardScaler, OneHotEncoder, Feature Engineering
+--------------------------+
          |
          v
+--------------------------+
|   Voting Classifier      | <-- RandomForest, GradientBoosting, XGBoost, LogisticRegression
+--------------------------+
          |
          v
+--------------------------+
|   Risk Prediction & Score| <-- Probability, Risk Categories, Threshold Optimization
+--------------------------+
          |
          v
+--------------------------+
|       AI Chatbot         | <-- LLM + RAG Memory + API Integration
+--------------------------+
          |
          v
+--------------------------+
| Personalized Recommendations | <-- Risk Explanation, Lifestyle Advice
+--------------------------+
```

#### 2️⃣ Chatbot Workflow Diagram
```
User Input (Question / Profile Data)
          |
          v
+--------------------+
|  Session State     | <-- Stores user profile, prior interactions
+--------------------+
          |
          v
+--------------------+
| Prefix Messages    | <-- System role + user profile + instructions
+--------------------+
          |
          v
+--------------------+
|   RAG Engine       | <-- Retrieves relevant context from Vector DB
+--------------------+
          |
          v
+--------------------+
|   LLM (Grok API)   | <-- Generates concise and context-aware response
+--------------------+
          |
          v
+--------------------+
|  Response Rendering| <-- Displayed on Streamlit Chat Interface
+--------------------+
```


## 🚀 Final Model Deployment


-- TO BE WRITTEN --



📌 Deployment includes:

Preprocessing inside pipeline
Calibrated probabilities
Threshold-applied risk classification
User-friendly Streamlit UI visualization

## Limitations

## 🔮 Future Work

- Add deeper clinical features
- Time-series health monitoring
- Include ECG/medical imaging features
- Build API and mobile app
- Continuous model monitoring

## 🏁 Conclusion

* Achieved better recall with good precision trade-off
* Developed a fully interpretable and deployable model
* Improved heart risk detection using advanced ML techniques
* Used SHAP for transparency
* Final model suitable for healthcare assistance tools
            """)