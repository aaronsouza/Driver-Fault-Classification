import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
classification_report, roc_curve, precision_recall_curve, auc
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("Accident.csv", parse_dates=['Crash Date/Time'])

# Create a new column for report category
df['Report Category'] = df['Report Number'].str[:1]

# Drop unnecessary columns
df.drop(['Off-Road Description', 'Municipality', 'Related Non-Motorist', 'Non-Motorist Substance Abuse',
'Circumstance', 'Person ID', 'Vehicle ID', 'Report Number'], axis=1, inplace=True)

# Extract date-time features
df['Crash_day'] = df['Crash Date/Time'].dt.day
df['Crash_hour'] = df['Crash Date/Time'].dt.hour
df['Crash_month'] = df['Crash Date/Time'].dt.month
df['Crash_year'] = df['Crash Date/Time'].dt.year
df['Crash_day_of_week'] = df['Crash Date/Time'].dt.dayofweek
df.drop('Crash Date/Time', axis=1, inplace=True)

# Process location data
df[['Location_x', 'Location_y']] = df['Location'].str.replace(',', ' ').str.split(' ', expand=True).astype(float)
df.drop('Location', axis=1, inplace=True)

# Fill missing values
for col in df.select_dtypes(include=[np.number]).columns:
df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=[object]).columns:
df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical variables
for col in ['Road Name', 'Cross-Street Name', 'Vehicle Make', 'Vehicle Model']:
df[f'enc_{col}'] = df.groupby(col).transform('size') / len(df)
df.drop(['Road Name', 'Cross-Street Name', 'Vehicle Make', 'Vehicle Model'], axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Prepare target variable
df['Fault'] = df['Fault'].apply(lambda x: 1 if x == 1 else 0)

# Split the data
X, y = df.drop('Fault', axis=1), df['Fault']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Train the model
xgb_model = XGBClassifier(random_state=101, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # Probability of the positive class

# Calculate metrics
metrics = {'Accuracy': accuracy_score(y_test, y_pred),
'Precision': precision_score(y_test, y_pred),
'Recall': recall_score(y_test, y_pred),
'F1-Score': f1_score(y_test, y_pred)}

# Print metrics
for k, v in metrics.items():
print(f"{k}: {v:.4f}")

print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(12, 6))

# ROC Curve subplot
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()

# Get feature importances
importances = xgb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Get the top 10 features
top_10_features = feature_importance_df.head(10)

# Plotting the top 10 features
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Features')
plt.gca().invert_yaxis() # Invert y-axis to have the most important feature at the top
plt.show()
