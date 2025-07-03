# Task 3: Decision Tree Classifier on Bank Marketing Data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (simulated)
data = {
    'age': [30, 40, 35, 28, 50, 45, 32, 60, 41, 33],
    'job': ['admin.', 'technician', 'blue-collar', 'student', 'retired', 'services', 'admin.', 'retired', 'technician', 'student'],
    'marital': ['married', 'single', 'married', 'single', 'married', 'divorced', 'single', 'married', 'divorced', 'single'],
    'education': ['secondary', 'tertiary', 'secondary', 'tertiary', 'primary', 'secondary', 'tertiary', 'primary', 'tertiary', 'secondary'],
    'balance': [1000, 2000, 1500, 100, 3000, 1200, 1100, 4000, 1800, 90],
    'previous': [0, 1, 0, 2, 1, 0, 0, 1, 3, 0],
    'y': ['no', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('y', axis=1), drop_first=True)
y = df['y'].map({'no': 0, 'yes': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.3, random_state=42)

# Train decision tree
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize decision tree
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=df_encoded.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Classifier")
plt.show()
