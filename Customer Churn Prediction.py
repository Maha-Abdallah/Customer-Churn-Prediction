#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'your_dataset.csv' with the actual dataset path)
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess the data
# Assuming 'Churn' is the target variable, and other columns are features
X = data.drop('Churn', axis=1)
y = data['Churn']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)


# In[2]:




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)
logreg_predictions = logreg_model.predict(X_test_scaled)

# Evaluate models
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, logreg_predictions)}")
print("Classification Report:")
print(classification_report(y_test, logreg_predictions))


# In[3]:


# Decision Trees
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
dt_predictions = dt_model.predict(X_test_scaled)

print("\nDecision Trees:")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions)}")
print("Classification Report:")
print(classification_report(y_test, dt_predictions))


# In[4]:


# Neural Networks (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
mlp_predictions = mlp_model.predict(X_test_scaled)




print("\nNeural Networks (MLP):")
print(f"Accuracy: {accuracy_score(y_test, mlp_predictions)}")
print("Classification Report:")
print(classification_report(y_test, mlp_predictions))

