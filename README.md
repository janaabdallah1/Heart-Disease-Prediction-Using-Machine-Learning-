# Heart Disease Prediction Using Machine Learning

A complete machine learning project for predicting **heart disease** using demographic, cardiovascular, ECG, and exercise-related patient information.

The project covers the full machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, feature selection, model comparison, hyperparameter tuning, evaluation, and deployment using **Streamlit**.

## Developed By

**Jana Mahmoud Abdalla**
**ID:** 231001245

**Course:** CBIO313 – Data Mining and Machine Learning
**Supervisor:** Dr. Mohamed EL-Sayeh
**Summer 2026**

---

## Dataset

The project uses the **Heart Failure Prediction Dataset** available on Kaggle.

**Dataset:** Heart Failure Prediction Dataset
**Author:** fedesoriano
**File:** `heart.csv`

The dataset contains:

* **918 patient records**
* **11 predictor variables**
* **1 binary target variable:** `HeartDisease`

### Features

* `Age`
* `Sex`
* `ChestPainType`
* `RestingBP`
* `Cholesterol`
* `FastingBS`
* `RestingECG`
* `MaxHR`
* `ExerciseAngina`
* `Oldpeak`
* `ST_Slope`

### Target

`HeartDisease`

* `0` = No Heart Disease
* `1` = Heart Disease

---

## Project Workflow

The project follows an end-to-end machine learning pipeline:

1. Data loading and inspection
2. Missing and invalid value detection
3. Data cleaning
4. Exploratory Data Analysis
5. Feature engineering
6. Train-test splitting
7. Numerical feature scaling
8. Categorical feature encoding
9. Feature selection using `SelectKBest`
10. Training multiple machine learning models
11. Model performance comparison
12. Hyperparameter tuning using `GridSearchCV`
13. Final model evaluation
14. Model saving using `joblib`
15. Deployment using Streamlit

---

## Data Preprocessing

The raw dataset contained no missing values or duplicated records. However, invalid zero values were detected in some clinical measurements.

The preprocessing workflow included:

* Checking missing values
* Checking duplicated rows
* Treating invalid zero values in `RestingBP` and `Cholesterol`
* Replacing invalid values using the median
* Scaling numerical variables using `StandardScaler`
* Encoding categorical variables using `OneHotEncoder`

---

## Feature Engineering

Additional features were generated to improve data interpretation:

* **Age Group**
* **Blood Pressure Category**
* **Cholesterol Category**
* **Percentage of Age-Predicted Maximum Heart Rate**
* **Exercise Capacity Category**

---

## Machine Learning Models

Six classification algorithms were trained and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Gradient Boosting

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC

---

## Final Model

After model comparison, **Random Forest** was optimized using `GridSearchCV` with five-fold cross-validation.

### Final Tuned Random Forest Performance

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **0.897** |
| Precision | **0.888** |
| Recall    | **0.931** |
| F1-Score  | **0.909** |
| ROC AUC   | **0.927** |

The high recall indicates that the final model successfully detects a large proportion of patients belonging to the heart-disease class.

---

## Visualizations

The project includes several exploratory and model-evaluation plots, including:

* Target distribution
* Age distribution
* Cholesterol distribution
* Resting blood pressure distribution
* Maximum heart rate distribution
* Age vs Heart Disease
* Cholesterol vs Heart Disease
* Chest Pain Type vs Heart Disease
* Exercise-Induced Angina vs Heart Disease
* Sex vs Heart Disease
* Correlation heatmap
* Model performance comparison
* Confusion matrix
* ROC curve

---

## Streamlit Application

An interactive Streamlit application was developed to allow users to enter patient information and obtain a heart disease risk prediction.

The application accepts:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

The app automatically performs the same feature engineering used during model training and displays:

* Predicted heart disease class
* Estimated probability
* Automatically generated health categories
* Patient input summary

---

## Running the Streamlit App

### 1. Clone or download the repository

Open the project folder in your terminal.

```bash
cd Jana_Mahmoud_Abdalla_231001245_Heart_Disease_Project
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized:

```bash
python -m streamlit run app.py
```

The application should open automatically in your browser at:

```text
http://localhost:8501
```

---

## Requirements

The main Python libraries used are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

---

## Repository Structure

```text
Jana_Mahmoud_Abdalla_231001245_Heart_Disease_Project/
│
├── Jana_Mahmoud_Abdalla_231001245_Project.ipynb
├── Jana_Mahmoud_Abdalla_231001245_Final_Report.pdf
├── Jana_Mahmoud_Abdalla_231001245_Final_Report.docx
│
├── heart.csv
├── Cleaned_Heart_Disease_Data.csv
├── X_features.csv
├── y_target.csv
├── model_comparison_results.csv
│
├── best_heart_disease_model.pkl
├── app.py
├── requirements.txt
│
├── 01_target_distribution.png
├── 02_age_distribution.png
├── 03_cholesterol_distribution.png
├── 04_restingbp_distribution.png
├── 05_maxhr_distribution.png
├── 06_age_vs_heartdisease.png
├── 07_cholesterol_vs_heartdisease.png
├── 08_chestpain_vs_heartdisease.png
├── 09_angina_vs_heartdisease.png
├── 10_sex_vs_heartdisease.png
├── 11_correlation_heatmap.png
├── 12_model_comparison.png
├── 13_confusion_matrix.png
└── 14_roc_curve.png
```

---

## Disclaimer

This project was developed for **educational purposes** as part of a Data Mining and Machine Learning course.

The prediction model is intended only as a demonstration of machine learning techniques and should **not be used as a substitute for professional medical diagnosis, clinical testing, or medical advice**.
