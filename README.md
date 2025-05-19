#Heart Disease Prediction using Machine Learning

Objective
To predict the presence of heart disease in patients using multiple machine learning classifiers and compare their performance.

### 1. **Dataset Overview**
Dataset: heart_disease_uci.csv
Columns:
  Numerical: age, trestbps, chol, thalch, oldpeak
  Categorical: sex, cp, fbs, restecg, exang, slope, etc.
  Target: num (converted to binary_num → 0: no disease, 1: disease)

### 2. **Data Preprocessing**

- **Missing Values Handling**:
    - Dropped columns with >50% missing values.
    - Filled missing values in:
        - Numerical columns → median
        - Categorical columns → mode
- **Label Encoding**: Converted categorical variables into numerical format.
- **Scaling**: Standardized selected numerical features using `StandardScaler`.
- **Target Transformation**: Converted `num` column into binary classification (`binary_num`).

### 3. ** Train-Test Split**
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
Accuracy Comparison (Visualized)
A bar plot was generated comparing model accuracies. Typical output:
Model

### 6. **Testing**
A patient’s clinical data was evaluated using an ensemble of well-trained machine learning model. 
Each model analyzed the input through a classification algorithm designed to predict the presence or absence of heart disease.

The results from all models were aggregated using majority voting, a form of hard voting ensemble learning.
Each model provided a binary prediction:
- 0 indicating no heart disease
- 1 indicating presence of heart disease
The final decision was based on the most frequently occurring prediction.
