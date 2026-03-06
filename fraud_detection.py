import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# ==============================
# 1 Load Dataset
# ==============================

data = pd.read_csv("dataset/creditcard.csv")

print("First rows of dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nDataset Statistics:")
print(data.describe())

print("\nClass Distribution:")
print(data['Class'].value_counts())

print("\nDataset shape:", data.shape)


# ==============================
# 2 Visualization
# ==============================

sns.countplot(x='Class', data=data)
plt.title("Fraud vs Normal Transactions")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()


# ==============================
# 3 Fraud Analysis
# ==============================

fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print("\nFraud Transactions:", len(fraud))
print("Normal Transactions:", len(normal))

fraud_percentage = (len(fraud) / len(data)) * 100
print("Fraud Percentage:", fraud_percentage)


# ==============================
# 4 Correlation Heatmap
# ==============================

plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# ==============================
# 5 Data Preprocessing
# ==============================

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Time'], axis=1)

print("\nData preprocessing completed")


# ==============================
# 6 Handle Imbalanced Dataset
# ==============================

fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

normal_sample = normal.sample(len(fraud))

balanced_data = pd.concat([fraud, normal_sample], axis=0)

print("\nBalanced dataset:")
print(balanced_data['Class'].value_counts())


# ==============================
# 7 Split Dataset
# ==============================

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==============================
# 8 Train Machine Learning Model
# ==============================

model = RandomForestClassifier()

model.fit(X_train, y_train)

print("\nModel training completed")


# ==============================
# 9 Model Prediction
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 10 Model Evaluation
# ==============================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==============================
# 11 Save the Model
# ==============================

joblib.dump(model, "fraud_model.pkl")

print("\nModel saved successfully as fraud_model.pkl")