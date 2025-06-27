
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

# Load dataset
data = pd.read_csv("breast_cancer_cleaned.csv")
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

# Metrics
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Threshold tuning at 0.3
threshold = 0.3
y_pred_thresh = (y_proba >= threshold).astype(int)
print("Tuned Confusion Matrix:", confusion_matrix(y_test, y_pred_thresh))
print("Tuned Precision:", precision_score(y_test, y_pred_thresh))
print("Tuned Recall:", recall_score(y_test, y_pred_thresh))
