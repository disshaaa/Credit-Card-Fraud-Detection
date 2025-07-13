# 💳 Credit Card Fraud Detection using Tuned Random Forest Classifier

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. Given the **high class imbalance** between legitimate and fraudulent transactions, we use **resampling techniques** and **hyperparameter tuning** to improve model performance — especially for detecting frauds (minority class).

We used a **Random Forest Classifier**, fine-tuned using `RandomizedSearchCV`, and evaluated performance with standard classification metrics.

---

## 📂 Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**:
  - 30 anonymized features (V1 to V28), `Amount`, and `Time`
  - `Class` label:  
    - `0`: Legitimate  
    - `1`: Fraudulent

---

## ⚙️ Methods Used

### 🧪 1. Exploratory Data Analysis (EDA)
- **Class Distribution**: Highly imbalanced (frauds ≈ 0.17%).
- **Transaction Amount Distribution**: Right-skewed, most values are small.

### 🧹 2. Preprocessing
- Scaled the `Amount` feature using `StandardScaler`
- Stratified train-test split to maintain fraud ratio
- Used **Random Over Sampling** on the training set to balance classes

### 🔍 3. Model & Tuning

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=1,  # Limited due to resource constraints
    scoring='f1',
    cv=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_resampled, y_train_resampled)
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)
```

## 📊 Results

### 🔢 1. Confusion Matrix
| Actual / Predicted | Predicted: 0 (Legit) | Predicted: 1 (Fraud) |
| ------------------ | -------------------- | -------------------- |
| **Actual: 0**      | 56864                | 13                   |
| **Actual: 1**      | 17                   | 81                   |

### 🧾 2. Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.75      0.82      0.78        98

    accuracy                           1.00     56962
    macro avg       0.87      0.91      0.89     56962
    weighted avg       1.00      1.00      1.00     56962

### 📌 3. Interpretation
- Recall (fraud) = 0.82 → Caught 82% of all actual frauds
- Precision (fraud) = 0.75 → 75% of predicted frauds were correct
- F1-Score (fraud) = 0.78 → Balanced measure of fraud detection
- Accuracy = 100% → But should be interpreted carefully due to class imbalance
- Weighted average = 1.00 → Overall performance is excellent due to dominance of class 0

## 🧠 Future Enhancements
- Try SMOTE or ADASYN instead of RandomOverSampler
- Use more iterations (n_iter > 1) in RandomizedSearchCV
- Experiment with other classifiers like XGBoost or LightGBM
- Perform feature selection to simplify model and reduce overfitting

  
