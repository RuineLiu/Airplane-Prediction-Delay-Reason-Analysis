# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import joblib

# 1. Load dataset from a Parquet file
# (Replace 'path/to/dataset.parquet' with the actual file path)
df = pd.read_parquet('path/to/dataset.parquet')

# 2. Data cleaning: remove rows with NaN target (DepDel15 is the target variable)
target = 'DepDel15'
df = df[df[target].notna()].copy()  # Remove samples with NaN in target
# Convert target to int (originally 0.0/1.0 floats) for consistency
df[target] = df[target].astype(int)

# 3. Feature removal to avoid data leakage and time travel
# Define keywords of leakage/time-travel features to drop
leak_keywords = [
    "CRSDepTime", "DepTime", "DepDelay", "DepDel15",  # DepDel15 is target
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15",
    "Cancelled", "Diverted",
    "DepTime_minutes", "DepTimestamp", 
    "ArrTime_minutes", "ArrTimestamp",
    "origin_delay_rate_past2h", "origin_arr_delay_avg_past2h",
    "origin_late_arrivals_past2h"
]
# Select feature columns that do not contain any of the above keywords
feature_cols = [col for col in df.columns 
                if col != target and not any(kw in col for kw in leak_keywords)]
X = df[feature_cols]
y = df[target]

# 4. Split data into training and test sets (consider stratification due to class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Handle missing values for numerical and categorical features
# Identify categorical columns (object or categorical dtype) and numerical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing values in training data
for col in cat_cols:
    X_train[col].fillna('Missing', inplace=True)   # fill missing categorical with placeholder
for col in num_cols:
    median_val = X_train[col].median()
    X_train[col].fillna(median_val, inplace=True)  # fill missing numeric with median of train

# Fill missing values in test data using training set statistics/placeholders
for col in cat_cols:
    X_test[col].fillna('Missing', inplace=True)
for col in num_cols:
    X_test[col].fillna(X_train[col].median(), inplace=True)  # use train median for consistency

# 6. Encode categorical features using Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on training data and transform both train and test
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    # If there are categories in test not seen in train, handle them by extending classes
    unseen = set(X_test[col].astype(str)) - set(le.classes_)
    if unseen:
        # Add unseen categories to classes (assign new labels)
        # Extend the classes of label encoder
        le.classes_ = np.append(le.classes_, list(unseen))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# 7. Train LightGBM classification model (consider class imbalance)
# Use class_weight="balanced" to automatically handle class imbalance:contentReference[oaicite:0]{index=0}
model = lgb.LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# 8. Save the trained model to disk using joblib
joblib.dump(model, 'lightgbm_model.joblib')

# 9. Evaluate model on the test set with various metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability for class 1

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
