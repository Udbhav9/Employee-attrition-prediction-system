import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('C:\\Users\\Vishal Mathur\\OneDrive\\Documents\\sales data\\ibm employee attrition data.csv')

print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})

df.drop(columns=['EmployeeNumber'], inplace=True)

attrition_rate = df['Attrition'].mean() * 100
print(f'Attrition Rate: {attrition_rate:.2f}%')

print(df.select_dtypes(include=['object']).columns)
df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'], drop_first=True)
print(df.info())

# Separate features and target variable
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Step 6: Split the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))
