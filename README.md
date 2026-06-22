
---

## 📂 TASK 2: Disease Prediction

**File:** `README.md`

```markdown
# Disease Prediction - Heart Disease Classification

## 📌 Project Overview
This project predicts the possibility of heart disease in patients using medical data. Multiple machine learning algorithms are compared to find the best performing model.

## 📊 Results
- **Best Model:** SVM (Support Vector Machine)
- **Accuracy:** 90.16%
- **Precision:** 0.90
- **Recall:** 0.90
- **F1-Score:** 0.90

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 85.25% | 0.85 | 0.85 | 0.85 |
| SVM | **90.16%** | **0.90** | **0.90** | **0.90** |
| Random Forest | 85.25% | 0.85 | 0.85 | 0.85 |
| XGBoost | 86.89% | 0.87 | 0.87 | 0.87 |

## 🛠️ Technologies Used
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost

## 📂 Dataset
**Heart Disease Dataset (UCI ML Repository)**
- Source: UCI Machine Learning Repository
- Features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise induced angina, ST depression, slope, number of vessels, thalassemia
- Target: 0 = No Heart Disease, 1 = Heart Disease

## 🚀 How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost

# Run the code
python disease_prediction.py
