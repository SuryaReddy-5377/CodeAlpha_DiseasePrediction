"""
TASK 2: Disease Prediction - Diabetes Classification
CodeAlpha Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("="*60)
print("🩺 DIABETES PREDICTION - TASK 2")
print("="*60)

# ============================================
# 1. LOAD DATASET
# ============================================
print("\n📥 Loading Diabetes Dataset...")

# Load from UCI (Public dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=columns)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# 2. EXPLORE DATA
# ============================================
print("\n📊 Data Overview:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())

# Check for missing values (zeros in medical data)
print("\n🔍 Checking for missing values (zeros in medical features):")
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zeros = (df[col] == 0).sum()
    print(f"  {col}: {zeros} zeros")

# ============================================
# 3. DATA PREPROCESSING
# ============================================
print("\n🔄 Data Preprocessing...")

# Replace zeros with median for medical columns
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    median = df[col].median()
    df[col] = df[col].replace(0, median)
    print(f"  ✅ Replaced zeros in {col} with median: {median}")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")
print(f"📊 Class distribution:")
print(f"  No Diabetes (0): {sum(y==0)} patients")
print(f"  Diabetes (1): {sum(y==1)} patients")

# ============================================
# 4. SPLIT DATA
# ============================================
print("\n📊 Splitting data: 80% Train, 20% Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# ============================================
# 5. SCALE FEATURES
# ============================================
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Feature scaling complete!")

# ============================================
# 6. TRAIN MODELS
# ============================================
print("\n🚀 Training Models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"  ✅ {name} Accuracy: {accuracy*100:.2f}%")

# ============================================
# 7. FIND BEST MODEL
# ============================================
print("\n" + "="*60)
print("📊 MODEL COMPARISON")
print("="*60)

best_model = max(results, key=results.get)
print("\n🔍 Results Summary:")
for name, acc in results.items():
    print(f"  {name}: {acc*100:.2f}%")

print(f"\n🏆 BEST MODEL: {best_model} with {results[best_model]*100:.2f}% accuracy")

# ============================================
# 8. DETAILED ANALYSIS OF BEST MODEL
# ============================================
print("\n" + "="*60)
print(f"📊 DETAILED ANALYSIS: {best_model}")
print("="*60)

# Train best model
if best_model == 'Logistic Regression':
    final_model = LogisticRegression(random_state=42)
elif best_model == 'Random Forest':
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    final_model = SVC(random_state=42)

final_model.fit(X_train_scaled, y_train)
y_pred_final = final_model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print(f"\n📊 Confusion Matrix:")
print(f"  True Negatives: {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives: {cm[1][1]}")

# Classification Report
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_final, 
                           target_names=['No Diabetes', 'Diabetes']))

# ============================================
# 9. VISUALIZATIONS
# ============================================
print("\n📊 Generating Visualizations...")

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix - {best_model}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance (for Random Forest)
if best_model == 'Random Forest':
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance - {best_model}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Feature Importance Ranking:")
    for i, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']*100:.1f}%")

# ============================================
# 10. PREDICT ON NEW DATA
# ============================================
print("\n" + "="*60)
print("🔍 TEST WITH NEW PATIENT DATA")
print("="*60)

# Example patient data
new_patient = np.array([[5, 120, 80, 25, 100, 35, 0.5, 30]])  # [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]
new_patient_scaled = scaler.transform(new_patient)

prediction = final_model.predict(new_patient_scaled)
probability = final_model.predict_proba(new_patient_scaled) if hasattr(final_model, 'predict_proba') else None

print("\n📋 Patient Data:")
print(f"  Pregnancies: 5")
print(f"  Glucose: 120")
print(f"  Blood Pressure: 80")
print(f"  Skin Thickness: 25")
print(f"  Insulin: 100")
print(f"  BMI: 35")
print(f"  Diabetes Pedigree: 0.5")
print(f"  Age: 30")

if probability is not None:
    print(f"\n📊 Prediction:")
    print(f"  No Diabetes: {probability[0][0]*100:.1f}%")
    print(f"  Diabetes: {probability[0][1]*100:.1f}%")

if prediction[0] == 0:
    print("\n✅ RESULT: No Diabetes Detected (Low Risk)")
else:
    print("\n⚠️ RESULT: Diabetes Detected (High Risk)")

print("\n" + "="*60)
print("🎉 TASK 2 COMPLETED SUCCESSFULLY!")
print("="*60)
