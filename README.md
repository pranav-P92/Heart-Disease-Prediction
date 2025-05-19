# Heart Disease Prediction using Machine Learning



**Objective**

To predict the presence of heart disease in patients using multiple machine learning classifiers and compare their performance.

### 1. **Dataset Overview**
- **Dataset:** heart_disease_uci.csv
- **Column Descriptions:**
    - **id** (Unique id for each patient)
    - **age** (Age of the patient in years)
    - **origin** (place of study)
    - **sex** (Male/Female)
    - **cp** chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
    - **trestbps** resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
    - **chol** (serum cholesterol in mg/dl)
    - **fbs** (if fasting blood sugar > 120 mg/dl)
    - **restecg** (resting electrocardiographic results)
        - **Values**: [normal, stt abnormality, lv hypertrophy]
    - **thalach**: maximum heart rate achieved
    - **exang**: exercise-induced angina (True/ False)
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slope**: the slope of the peak exercise ST segment
    - **ca**: number of major vessels (0-3) colored by fluoroscopy
    - **thal**: [normal; fixed defect; reversible defect]
    - **num**: the predicted attribute
- **Numerical:** age, trestbps, chol, thalch, oldpeak
- **Categorical:** sex, cp, fbs, restecg, exang, slope, etc.
- **Target:** num (converted to binary_num → 0: no disease, 1: disease)

### 2. **Data Preprocessing**

- **Missing Values Handling**:
    - Dropped columns with >50% missing values.
    - Filled missing values in:
        - Numerical columns → median
        - Categorical columns → mode
- **Label Encoding**: Converted categorical variables into numerical format.
- **Scaling**: Standardized selected numerical features using `StandardScaler`.
- **Target Transformation**: Converted `num` column into binary classification (`binary_num`).

### 3. **Train-Test Split**
Ensures balanced distribution of classes in both sets.
  - Training data : 80%
  - Testing data  : 20%


 ### 4. **Model Training & Evaluation**

Eight models were trained and evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Extra Trees
- AdaBoost
- XGBoost
- LightGBM

All models were evaluated using:
- `accuracy_score`
- `classification_report`
- `confusion_matrix`


### 5. **Results**
Accuracy Comparison (Visualized).

  - Logistic Regression Accuracy: 0.831
  - Random Forest Accuracy      : 0.836
  - Gradient Boosting Accuracy  : 0.831
  - KNN Accuracy                : 0.842
  - Extra Trees Accuracy        : 0.826
  - AdaBoost Accuracy           : 0.836
  - XGBoost  Accuracy           : 0.836
  - LightGBM  Accuracy          : 0.815


**Logistic Regression Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.82      0.79      0.81        82
               1       0.84      0.86      0.85       102

        accuracy                           0.83       184
       macro avg       0.83      0.83      0.83       184
    weighted avg       0.83      0.83      0.83       184


**Random Forest Classification Report:**

                     precision    recall  f1-score   support
    
               0       0.83      0.79      0.81        82
               1       0.84      0.87      0.86       102
    
         accuracy                           0.84       184
        macro avg       0.84      0.83      0.83       184 
     weighted avg       0.84      0.84      0.84       184


**Gradient Boosting Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.84      0.77      0.80        82
               1       0.83      0.88      0.85       102
    
        accuracy                           0.83       184
       macro avg       0.83      0.83      0.83       184
    weighted avg       0.83      0.83      0.83       184


**KNN Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.82      0.83      0.82        82
               1       0.86      0.85      0.86       102
    
        accuracy                           0.84       184
       macro avg       0.84      0.84      0.84       184
    weighted avg       0.84      0.84      0.84       184


**Extra Trees Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.81      0.79      0.80        82
               1       0.84      0.85      0.84       102
    
        accuracy                           0.83       184
       macro avg       0.82      0.82      0.82       184
    weighted avg       0.83      0.83      0.83       184


**AdaBoost Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.83      0.79      0.81        82
               1       0.84      0.87      0.86       102
    
        accuracy                           0.84       184
       macro avg       0.84      0.83      0.83       184
    weighted avg       0.84      0.84      0.84       184


**XGBoost Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.82      0.80      0.81        82
               1       0.85      0.86      0.85       102
    
        accuracy                           0.84       184
       macro avg       0.84      0.83      0.83       184
    weighted avg       0.84      0.84      0.84       184


**LightGBM Classification Report:**

                   precision    recall  f1-score   support
    
               0       0.79      0.79      0.79        82
               1       0.83      0.83      0.83       102
    
        accuracy                           0.82       184
       macro avg       0.81      0.81      0.81       184
    weighted avg       0.82      0.82      0.82       184

### 6. **Testing**
A patient’s clinical data was evaluated using an ensemble of well-trained machine learning models. Each model analyzed the input through a classification algorithm designed to predict the presence or absence of heart disease.

The results from all models were aggregated using **majority voting**, a form of **hard voting ensemble learning**.
Each model provided a binary prediction:
  - **0** indicating **no heart disease.**
  - **1** indicating **presence of heart disease.**

The final decision was made based on the most frequently occurring prediction, indicating whether heart disease is present or absent.

