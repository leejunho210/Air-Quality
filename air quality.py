# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = '/path/to/dataset.csv'  # Replace with the actual path
data = pd.read_csv(file_path)

# Assign column names
columns = ['MQ2', 'MQ9', 'MQ135', 'MQ137', 'MQ138', 'MG-811', 'label']
data.columns = columns

# Add a simulated time index for time-series analysis
data['time'] = np.arange(len(data))

# Split the data into features (X) and target (y)
X = data[['MQ2', 'MQ9', 'MQ135', 'MQ137', 'MQ138', 'MG-811']]
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Sensor data trends over time
plt.figure(figsize=(10, 6))
for sensor in ['MQ2', 'MQ9', 'MQ135', 'MQ137', 'MQ138', 'MG-811']:
    plt.plot(data['time'], data[sensor], label=sensor)
plt.legend()
plt.title('Sensor Data Trends Over Time')
plt.xlabel('Time Index')
plt.ylabel('Sensor Values')
plt.show()

# Feature Importance Visualization
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Output Results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("Best Parameters:", grid_search.best_params_)