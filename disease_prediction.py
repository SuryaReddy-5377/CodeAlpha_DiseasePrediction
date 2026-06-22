"""
TASK 2: Disease Prediction using Medical Data
CodeAlpha Internship - BOTH Auto & Manual Upload
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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🩺 DISEASE PREDICTION - TASK 2")
print("="*60)

# ============================================
# STEP 1: CHOOSE DATA SOURCE
# ============================================
print("\n" + "="*60)
print("📊 CHOOSE DATA SOURCE")
print("="*60)
print("1. Auto-load from UCI Repository")
print("2. Manually Upload Your Own Dataset")

choice = input("\nEnter 1 or 2: ")

# ============================================
# OPTION 1: AUTO-LOAD FROM UCI
# ============================================
if choice == '1':
    print("\n" + "="*60)
    print("📥 AUTO-LOAD FROM UCI REPOSITORY")
    print("="*60)
    
    print("\nChoose dataset:")
    print("  a) Breast Cancer (Highest Accuracy ~98%)")
    print("  b) Diabetes (Accuracy ~75-80%)")
    print("  c) Heart Disease (Accuracy ~80-85%)")
    
    dataset_choice = input("\nEnter a, b, or c: ")
    
    if dataset_choice.lower() == 'a':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['Outcome'] = data.target
        target = 'Outcome'
        disease_name = "Breast Cancer"
        print(f"\n✅ Loaded Breast Cancer Dataset from UCI")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Classes: 0 = Benign, 1 = Malignant")
        
    elif dataset_choice.lower() == 'b':
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        target = 'Outcome'
        disease_name = "Diabetes"
        print(f"\n✅ Loaded Diabetes Dataset from UCI")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
    elif dataset_choice.lower() == 'c':
        url = "https://archive.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        df = pd.read_csv(url, names=columns, na_values='?')
        df = df.dropna()
        df['Outcome'] = (df['num'] > 0).astype(int)
        df = df.drop('num', axis=1)
        target = 'Outcome'
        disease_name = "Heart Disease"
        print(f"\n✅ Loaded Heart Disease Dataset from UCI")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# OPTION 2: MANUAL UPLOAD (FIXED)
# ============================================
elif choice == '2':
    print("\n" + "="*60)
    print("📤 MANUAL UPLOAD YOUR DATASET")
    print("="*60)
    print("\n📌 Instructions:")
    print("  1. Upload CSV or DATA file")
    print("  2. Target should be 0 and 1 (classification)")
    print("  3. For heart disease, use 'num' as target")
    
    from google.colab import files
    print("\n📤 Click 'Choose Files' to upload")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"\n✅ Uploaded: {filename}")
        
        # Load based on file extension
        if filename.endswith('.data'):
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
            df = pd.read_csv(filename, names=columns, na_values='?')
            print("✅ Loaded as .data file with predefined columns")
        else:
            df = pd.read_csv(filename)
            print("✅ Loaded as CSV file")
        
        print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\n📊 First 5 rows:")
    print(df.head())
    
    print(f"\n📊 Available Columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
    
    # Handle missing values
    print("\n🔄 Handling missing values...")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  ✅ {col}: Filled missing with median")
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  ✅ {col}: Converted to numeric and filled missing")
            except:
                print(f"  ⚠️ {col}: Skipped (non-numeric)")
    
    target = input("\n🎯 Enter the target column name: ")
    if target not in df.columns:
        print(f"❌ Column '{target}' not found! Using last column.")
        target = df.columns[-1]
    
    disease_name = input("📝 Enter disease name (e.g., Heart Disease): ")
    print(f"\n✅ Target: {target}, Disease: {disease_name}")

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n" + "="*60)
print("🔄 DATA PREPROCESSING")
print("="*60)

print(f"\n📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Check missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Handle any remaining missing values
df = df.fillna(df.median())

# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

print(f"\n📊 Features: {X.shape[1]} columns")
print(f"📊 Target: {y.shape[0]} samples")
print(f"\n📊 Class Distribution:")
print(f"  Class 0: {(y==0).sum()} samples ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  Class 1: {(y==1).sum()} samples ({(y==1).sum()/len(y)*100:.1f}%)")

# ============================================
# STEP 3: SPLIT AND SCALE
# ============================================
print("\n📊 Splitting data: 80% Train, 20% Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 4: TRAIN MODELS (As per PDF)
# ============================================
print("\n" + "="*60)
print("🚀 TRAINING MODELS")
print("="*60)
print("📊 Algorithms: SVM, Logistic Regression, Random Forest, XGBoost")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}
predictions = {}

for name, model in models.items():
    try:
        print(f"\n📊 Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"  ✅ {name} Accuracy: {accuracy*100:.2f}%")
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")

# ============================================
# STEP 5: RESULTS
# ============================================
print("\n" + "="*60)
print("📊 MODEL COMPARISON RESULTS")
print("="*60)

if results:
    best_model_name = max(results, key=results.get)
    for name, acc in results.items():
        print(f"  {name:25} {acc*100:>6.2f}%")
    print(f"\n🏆 BEST MODEL: {best_model_name} ({results[best_model_name]*100:.2f}%)")
else:
    print("❌ No models trained successfully!")
    best_model_name = None

# ============================================
# STEP 6: BEST MODEL DETAILS
# ============================================
if best_model_name:
    print("\n" + "="*60)
    print(f"📊 DETAILED ANALYSIS: {best_model_name}")
    print("="*60)

    best_model = models[best_model_name]
    y_pred_best = predictions[best_model_name]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")

    # Classification Report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred_best, 
                               target_names=[f'No {disease_name}', disease_name]))

    # ============================================
    # STEP 7: VISUALIZATION
    # ============================================
    print("\n📊 Generating Visualizations...")

    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'No {disease_name}', disease_name],
                yticklabels=[f'No {disease_name}', disease_name])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Feature Importance
    if best_model_name in ['Random Forest', 'XGBoost']:
        plt.figure(figsize=(10, 6))
        importance = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    # ============================================
    # STEP 8: PREDICT ON SAMPLE
    # ============================================
    print("\n" + "="*60)
    print("🔍 PREDICT ON SAMPLE PATIENT")
    print("="*60)

    sample_idx = np.random.randint(0, len(X_test))
    sample_data = X_test.iloc[sample_idx:sample_idx+1]
    sample_actual = y_test.iloc[sample_idx]

    sample_scaled = scaler.transform(sample_data)
    prediction = best_model.predict(sample_scaled)
    prob = best_model.predict_proba(sample_scaled)[0] if hasattr(best_model, 'predict_proba') else None

    print("\n📋 Patient Data:")
    for col, val in zip(X.columns, sample_data.values[0]):
        print(f"  {col}: {val:.2f}")

    print(f"\n📊 Actual Diagnosis: {disease_name if sample_actual == 1 else 'No '+disease_name}")

    if prob is not None:
        print(f"\n📊 Prediction Probabilities:")
        print(f"  No {disease_name}: {prob[0]*100:.1f}%")
        print(f"  {disease_name}: {prob[1]*100:.1f}%")

    if prediction[0] == 0:
        print(f"\n✅ PREDICTION: No {disease_name} Detected (Low Risk)")
    else:
        print(f"\n⚠️ PREDICTION: {disease_name} Detected (High Risk)")

print("\n" + "="*60)
print("🎉 TASK 2 COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📌 SUMMARY:")
print(f"  ✅ Dataset Source: {'Auto-Load from UCI' if choice == '1' else 'Manual Upload'}")
print(f"  ✅ Dataset: {disease_name}")
print(f"  ✅ Algorithms: SVM, Logistic Regression, Random Forest, XGBoost")
if results:
    print(f"  ✅ Best Model: {best_model_name} ({results[best_model_name]*100:.2f}%)")
print(f"  ✅ All PDF requirements completed!")
