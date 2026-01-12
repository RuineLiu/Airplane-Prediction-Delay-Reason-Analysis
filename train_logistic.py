import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 1. Load dataset: training (2018), validation (2019), and test (2022)
train_file = "flights_weather_propagation_2018.parquet"
val_file = "flights_weather_propagation_2019.parquet"
test_file = "flights_weather_propagation_2022.parquet"
print("Loading data from Parquet files...")
df_train = pd.read_parquet(train_file)
df_val = pd.read_parquet(val_file)
df_test = pd.read_parquet(test_file)

# Separate features (X) and target (y)
target_col = "DepDel15"
X_train = df_train.drop(columns=[target_col]); y_train = df_train[target_col]
X_val = df_val.drop(columns=[target_col]); y_val = df_val[target_col]
X_test = df_test.drop(columns=[target_col]); y_test = df_test[target_col]

# Reset index for safety (not strictly necessary, but ensures alignment after transformations)
X_train.reset_index(drop=True, inplace=True); y_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True); y_val.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True); y_test.reset_index(drop=True, inplace=True)

# 2. Handle missing values
print("Handling missing values...")
# For numeric features: fill NaNs with median of the training set
num_cols = X_train.select_dtypes(include=[np.number]).columns
for col in num_cols:
    median = X_train[col].median()
    X_train[col].fillna(median, inplace=True)
    X_val[col].fillna(median, inplace=True)
    X_test[col].fillna(median, inplace=True)
# For categorical features: fill NaNs with 'missing' and also tag unknown categories as 'missing'
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    X_train[col].fillna("missing", inplace=True)
    X_val[col].fillna("missing", inplace=True)
    X_test[col].fillna("missing", inplace=True)
    # Replace categories in val/test not seen in train with 'missing'
    train_cats = set(X_train[col])
    X_val.loc[~X_val[col].isin(train_cats), col] = "missing"
    X_test.loc[~X_test[col].isin(train_cats), col] = "missing"

# 3. Encode categorical features using One-Hot Encoding
print("Encoding categorical features (one-hot)...")
enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_train_cat = enc.fit_transform(X_train[cat_cols])
X_val_cat = enc.transform(X_val[cat_cols])
X_test_cat = enc.transform(X_test[cat_cols])
# Get one-hot encoded column names for reference
onehot_cols = enc.get_feature_names_out(cat_cols)
# Convert encoded arrays back to DataFrame for easy concatenation
X_train_cat = pd.DataFrame(X_train_cat, columns=onehot_cols)
X_val_cat = pd.DataFrame(X_val_cat, columns=onehot_cols)
X_test_cat = pd.DataFrame(X_test_cat, columns=onehot_cols)
# Reset indices on the new DataFrames to align with original indexes
X_train_cat.reset_index(drop=True, inplace=True)
X_val_cat.reset_index(drop=True, inplace=True)
X_test_cat.reset_index(drop=True, inplace=True)

# Drop original categorical columns (since we have one-hot replacements)
X_train_num = X_train[num_cols].reset_index(drop=True)
X_val_num = X_val[num_cols].reset_index(drop=True)
X_test_num = X_test[num_cols].reset_index(drop=True)

# 4. Feature scaling for numeric features (Standardization)
print("Scaling numeric features...")
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols)
X_val_num_scaled = pd.DataFrame(scaler.transform(X_val_num), columns=num_cols)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols)

# Combine scaled numeric features and one-hot categorical features
X_train_proc = pd.concat([X_train_num_scaled, X_train_cat], axis=1)
X_val_proc = pd.concat([X_val_num_scaled, X_val_cat], axis=1)
X_test_proc = pd.concat([X_test_num_scaled, X_test_cat], axis=1)

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# 5. Train Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, solver="liblinear", verbose=1, random_state=42)
model.fit(X_train_proc, y_train)  # Training the model (verbose output will show optimization progress)

# 6. Save the trained model to file
model_path = "outputs/logistic_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 7. Evaluate on validation and test sets
print("Evaluating model on validation and test sets...")
# Predict class labels and probabilities
y_pred_val = model.predict(X_val_proc)
y_pred_test = model.predict(X_test_proc)
y_proba_val = model.predict_proba(X_val_proc)[:, 1]
y_proba_test = model.predict_proba(X_test_proc)[:, 1]

# Calculate evaluation metrics
acc_val = accuracy_score(y_val, y_pred_val)
prec_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)
auc_val = roc_auc_score(y_val, y_proba_val)
acc_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_proba_test)

# Print metrics to terminal
print(f"Validation Set – Accuracy: {acc_val:.4f}, Precision: {prec_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}, AUC: {auc_val:.4f}")
print(f"Test Set – Accuracy: {acc_test:.4f}, Precision: {prec_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}, AUC: {auc_test:.4f}")

# Save evaluation metrics to a CSV file
metrics_df = pd.DataFrame({
    "Accuracy": [acc_val, acc_test],
    "Precision": [prec_val, prec_test],
    "Recall": [recall_val, recall_test],
    "F1": [f1_val, f1_test],
    "AUC": [auc_val, auc_test]
}, index=["Validation", "Test"])
metrics_path = "outputs/logistic_metrics.csv"
metrics_df.to_csv(metrics_path)
print(f"Evaluation metrics saved to {metrics_path}")

# 8. Plot and save ROC curve for the model
fpr_val, tpr_val, _ = roc_curve(y_val, y_proba_val)
fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr_val, tpr_val, label=f"Validation (AUC={auc_val:.3f})")
plt.plot(fpr_test, tpr_test, label=f"Test (AUC={auc_test:.3f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend(loc="lower right")
roc_path = "outputs/logistic_roc_curve.png"
plt.savefig(roc_path)
plt.close()
print(f"ROC curve plot saved to {roc_path}")
